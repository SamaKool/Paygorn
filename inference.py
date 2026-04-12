#!/usr/bin/env python3

"""
Inference Script for FinAuditor
===================================
Refactored to strictly match the STDOUT FORMAT template.
"""

import asyncio
import os
import sys
import json
import re
import datetime
import traceback
import time
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# ── Project root on sys.path so `hft_auditor` .so and `models` are importable ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI
from pydantic import BaseModel

try:
    from hft_auditor_env import FinAuditorEnv as FinAuditorEnvironment
except ImportError:
    from server.fin_auditor_environment import FinAuditorEnvironment

from models import AuditorAction

class LLMResponse(BaseModel):
    reasoning: str
    decisions: List[int]

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME:   str = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN:     str = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("[DEBUG] CRITICAL: HF_TOKEN environment variable is missing.", flush=True)

TASK_ID:      str = os.getenv("TASK_ID", "anomaly_detection_hard")
BENCHMARK:    str = os.getenv("BENCHMARK", "fin_auditor")

if "easy" in TASK_ID.lower():
    _DEFAULT_MAX = 5
elif "medium" in TASK_ID.lower():
    _DEFAULT_MAX = 10
else:
    _DEFAULT_MAX = 20

MAX_STEPS:    int = int(os.getenv("MAX_STEPS", str(_DEFAULT_MAX)))
TEMPERATURE = 0.0
MAX_TOKENS = 1500
SUCCESS_SCORE_THRESHOLD = 0.5  # Need 50%+ to succeed

SYSTEM_PROMPT = textwrap.dedent(
    """
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
).strip()

_last_reasoning: str = ""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _build_user_prompt(step: int, features: list[list[float]]) -> str:
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

def _normalize_decisions(decisions: list[int], expected: int) -> list[int]:
    clamped = [1 if d >= 1 else 0 for d in decisions]
    clamped = clamped[:expected]
    while len(clamped) < expected:
        clamped.append(1) 
    return clamped

def get_model_message(client: OpenAI, step: int, features: list[list[float]]) -> list[int]:
    global _last_reasoning
    _last_reasoning = "Fallback triggered."
    user_prompt = _build_user_prompt(step, features)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False,
            )
            content = (completion.choices[0].message.content or "").strip()
            return _parse_llm_decisions(content, len(features))
        except Exception as exc:
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            time.sleep(1)

    fallback_decisions = []
    for row in features:
        if len(row) >= 4:
            risk_score = row[3]
            fallback_decisions.append(0 if risk_score < 0.30 else 1)
        else:
            fallback_decisions.append(1)
            
    return fallback_decisions

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        env = FinAuditorEnvironment()
        
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
                decisions = []
                action = AuditorAction(decisions=[])
                global _last_reasoning
                _last_reasoning = "Empty matrix."
            else:
                decisions = get_model_message(client, step, features)
                action = AuditorAction(decisions=decisions)

            obs = env.step(action)
            
            reward = float(obs.reward) if obs.reward is not None else 0.1
            done = obs.done
            error = getattr(obs, "error", None)

            rewards.append(reward)
            steps_taken = step
            
            action_str = ",".join(str(d) for d in decisions) if decisions else "none"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break
                
        # Calculate final score based on latest reward (as per Discord guidance clamped strictly)
        raw_score = rewards[-1] if rewards else 0.1
        score = max(0.01, min(0.99, float(raw_score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Execution error: {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            if env and hasattr(env, "close"):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        # Ensure fallback score if empty
        if not rewards:
            rewards = [0.1]
            score = 0.1
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
