#!/usr/bin/env python3

import os
import sys
import json
import re
import datetime
import traceback
import time
from typing import List

from dotenv import load_dotenv
load_dotenv()

# ── Project root on sys.path so `hft_auditor` .so and `models` are importable ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI
from pydantic import BaseModel, ValidationError

try:
    from hft_auditor_env import FinAuditorEnv as FinAuditorEnvironment
except ImportError:
    from server.fin_auditor_environment import FinAuditorEnvironment

from models import AuditorAction

class LLMResponse(BaseModel):
    reasoning: str
    decisions: List[int]

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN:     str = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("CRITICAL: HF_TOKEN environment variable is missing.")

TASK_ID:      str = os.getenv("TASK_ID", "anomaly_detection_hard")

if "easy" in TASK_ID.lower():
    _DEFAULT_MAX = 5
elif "medium" in TASK_ID.lower():
    _DEFAULT_MAX = 10
else:
    _DEFAULT_MAX = 20

MAX_STEPS:    int = int(os.getenv("MAX_STEPS", str(_DEFAULT_MAX)))

_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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

def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

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

_last_reasoning: str = ""

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

def _call_llm(step: int, features: list[list[float]]) -> list[int]:
    global _last_reasoning
    _last_reasoning = "Fallback triggered."
    user_prompt = _build_user_prompt(step, features)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return _parse_llm_decisions(content, len(features))
        except Exception as e:
            time.sleep(1)

    fallback_decisions = []
    for row in features:
        if len(row) >= 4:
            fallback_decisions.append(1 if row[3] >= 0.7 else 0)
        else:
            fallback_decisions.append(1)
            
    return fallback_decisions

def run_inference() -> None:
    episode_id: str = "unknown"
    total_reward: float = 0.0
    steps_completed: int = 0
    status: str = "SUCCESS"

    try:
        env = FinAuditorEnvironment()
        obs = env.reset()
        episode_id = getattr(env.state, 'episode_id', "test_run")

        start_payload = {
            "episode_id": episode_id, 
            "model": MODEL_NAME, 
            "difficulty": TASK_ID, 
            "max_steps": MAX_STEPS
        }
        print(f"[START] {json.dumps(start_payload)}", flush=True)

        for step_num in range(1, MAX_STEPS + 1):
            step_reward = 0.0  
            features = obs.features

            if not features:
                action = AuditorAction(decisions=[])
                _last_reasoning = "Empty matrix."
            else:
                decisions = _call_llm(step_num, features)
                action = AuditorAction(decisions=decisions)

            obs = env.step(action)
            # Apply safe floor fallback in inference just in case
            step_reward = obs.reward if obs.reward is not None else 0.01 
            total_reward += step_reward
            steps_completed = step_num

            # FIX: Used round(..., 4) to prevent collapsing small fractions into 0.00
            step_payload = {
                "step": step_num,
                "anomalies": len(features),
                "reward": round(float(step_reward), 4),
                "cumulative_reward": round(float(total_reward), 4),
                "done": bool(obs.done),
                "error": None,
                "reasoning": _last_reasoning[:120].replace('\n', ' ') + "...",
                "tp": getattr(env.state, 'last_tp', 0),
                "tn": getattr(env.state, 'last_tn', 0),
                "fp": getattr(env.state, 'last_fp', 0),
                "fn": getattr(env.state, 'last_fn', 0)
            }
            print(f"[STEP] {json.dumps(step_payload)}", flush=True)

            if obs.done:
                break

    except KeyboardInterrupt:
        status = "INTERRUPTED"
    except Exception as exc:
        status = "ERROR"
        traceback.print_exc(file=sys.stderr)

    avg_reward = total_reward / max(steps_completed, 1)
    
    # FIX: Used round(..., 4) for terminal payload outputs as well
    end_payload = {
        "total_reward": round(float(total_reward), 4), 
        "avg_reward": round(float(avg_reward), 4),
        "status": status
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

if __name__ == "__main__":
    run_inference()