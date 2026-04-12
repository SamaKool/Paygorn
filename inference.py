"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
import textwrap
import json
import re
import time
from typing import List, Optional
from pydantic import BaseModel

from openai import OpenAI

import sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from hft_auditor_env import FinAuditorEnv as FinAuditorEnvironment
except ImportError:
    from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"

TASK_ID = os.getenv("TASK_ID", "anomaly_detection_hard")
if "easy" in TASK_ID.lower():
    _DEFAULT_MAX = 5
elif "medium" in TASK_ID.lower():
    _DEFAULT_MAX = 10
else:
    _DEFAULT_MAX = 20

MAX_STEPS = int(os.getenv("MAX_STEPS", str(_DEFAULT_MAX)))
TEMPERATURE = 0.0
MAX_TOKENS = 1500

SYSTEM_PROMPT = """\
You are a Wall Street Compliance Auditor AI embedded inside a High-Frequency Trading audit engine.
You MUST think step-by-step in the 'reasoning' field before determining your action.

━━━ DECISION VALUES ━━━
You must output ONLY raw integers (0 or 1) in the array. NO strings. NO labels.
  1 : Confirmed anomaly.
  0 : Trade is clean.

━━━ EVALUATION DIRECTIVE ━━━
1. CRITICAL SIGNAL (FLAG): If risk_score > 0.60, output 1.
2. NOISE SIGNAL (PASS): If risk_score < 0.30, output 0.
3. AMBIGUITY (FLAG): If risk_score is in between, output 1 to be safe.

━━━ CRITICAL JSON FORMAT ━━━
You MUST respond with a valid JSON object. The decisions array MUST contain exactly the requested number of raw integers. NO trailing commas.
Example:
{"reasoning": "Trade 1 has high risk. Trade 2 is safe.", "decisions": [1, 0, 1]}
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, features: list[list[float]]) -> str:
    lines = [
        f"Step {step}: You have {len(features)} flagged trades to audit.",
        "",
        "Trade# | time_elapsed | price_delta | missing_freq | risk_score",
        "-------|--------------|-------------|--------------|----------",
    ]
    for i, row in enumerate(features):
        if len(row) >= 4:
            lines.append(f"  {i+1:3d}  |   {row[0]:8.4f}   |   {row[1]:7.4f}   |   {row[2]:8.4f}   |  {row[3]:7.4f}")
        else:
            lines.append(f"  {i+1:3d}  |   (malformed row: {row})")
    lines.append("")
    lines.append(f"Provide exactly {len(features)} decisions as a JSON object.")
    return "\n".join(lines)


class LLMResponse(BaseModel):
    reasoning: str
    decisions: List[int]

_last_reasoning: str = ""

def _normalize_decisions(decisions: list[int], expected: int) -> list[int]:
    clamped = [1 if d >= 1 else 0 for d in decisions]
    clamped = clamped[:expected]
    while len(clamped) < expected:
        clamped.append(1) 
    return clamped

def _parse_llm_decisions(content: str, expected_count: int) -> list[int]:
    global _last_reasoning
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```[\w]*\n?', '', stripped)
        stripped = re.sub(r'\n?```$', '', stripped.strip())

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "decisions" in parsed:
            response = LLMResponse(**parsed)
            _last_reasoning = response.reasoning
            return _normalize_decisions([int(d) for d in response.decisions], expected_count)
    except Exception:
        pass

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "decisions" in parsed:
            decisions = [int(d) for d in parsed["decisions"]]
            return _normalize_decisions(decisions, expected_count)
    except Exception:
        pass

    match = re.search(r'\[[\s\d,]+\]', content)
    if match:
        try:
            decisions = json.loads(match.group())
            return _normalize_decisions([int(d) for d in decisions], expected_count)
        except Exception:
            pass

    return [1] * expected_count

def get_model_message(client: OpenAI, step: int, features: list[list[float]]) -> list[int]:
    global _last_reasoning
    _last_reasoning = "Fallback triggered."
    user_prompt = build_user_prompt(step, features)
    max_retries = 3

    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            content = (completion.choices[0].message.content or "").strip()
            return _parse_llm_decisions(content, len(features))
        except Exception as exc:
            print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(1)
            
    fallback_decisions = []
    for row in features:
        if len(row) >= 4:
            # Matches SYSTEM_PROMPT: 1 if > 0.60, 0 if < 0.30, 1 if in between.
            risk_score = row[3]
            fallback_decisions.append(0 if risk_score < 0.30 else 1)
        else:
            fallback_decisions.append(1)
            
    return fallback_decisions


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = FinAuditorEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.10
    success = False

    log_start(task=TASK_ID, env="fin_auditor", model=MODEL_NAME)

    try:
        # Determine the correct task configuration dynamically based on TASK_ID
        if "easy" in TASK_ID.lower():
            from tasks.task1_easy import setup_env
            setup_env(env)
        elif "medium" in TASK_ID.lower():
            from tasks.task2_medium import setup_env
            setup_env(env)
        else:
            from tasks.task3_hard import setup_env
            setup_env(env)

        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            features = obs.features

            if not features:
                action = AuditorAction(decisions=[])
                global _last_reasoning
                _last_reasoning = "Empty matrix."
            else:
                decisions = get_model_message(client, step, features)
                action = AuditorAction(decisions=decisions)

            obs = env.step(action)

            base_reward = float(obs.reward) if obs.reward is not None else 0.1
            reward = float(max(0.01, min(0.99, base_reward)))
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = ",".join(str(d) for d in action.decisions) if action.decisions else "none"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if "easy" in TASK_ID.lower():
            from graders.grader_detection import EasyDetectionGrader
            grader = EasyDetectionGrader()
        elif "medium" in TASK_ID.lower():
            from graders.grader_classification import MediumClassificationGrader
            grader = MediumClassificationGrader()
        else:
            from graders.grader_fix import HardFixGrader
            grader = HardFixGrader()
            
        score = grader.grade(env.state)
        success = True

    except Exception as exc:
        print(f"[DEBUG] Inference failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if not rewards:
            rewards = [0.10]
            score = 0.10
            
        # Ensure absolutely no element is exactly 0.0 or 1.0 or outside the valid range.
        for i in range(len(rewards)):
            rewards[i] = float(max(0.01, min(0.99, rewards[i])))
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()