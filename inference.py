#!/usr/bin/env python3
"""
inference.py — LLM-powered HFT Compliance Auditor Agent
========================================================
Hackathon submission entry point.

Instantiates FinAuditorEnvironment directly (in-process, no HTTP round-trips)
and uses an OpenAI-compatible client with HF_TOKEN to drive the LLM decision
loop. Emits structured stdout logs in [START] / [STEP] / [END] format.

Required environment variables:
    API_BASE_URL  — OpenAI-compatible inference endpoint
    MODEL_NAME    — Model identifier (e.g., "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN      — HuggingFace token used as the API key

Optional:
    MAX_STEPS     — Number of RL steps to run (default: 10)
    TASK_ID       — Which task to run (default: anomaly_detection_easy)

Usage:
    API_BASE_URL=https://... MODEL_NAME=Qwen/... HF_TOKEN=hf_... python inference.py
"""

import os
import sys
import json
import re
import datetime
import traceback

# ── Project root on sys.path so `hft_auditor` .so and `models` are importable ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI
from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all sourced from environment variables per hackathon rules
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME:   str = os.environ["MODEL_NAME"]
HF_TOKEN:     str = os.environ["HF_TOKEN"]          # used as OpenAI api_key
MAX_STEPS:    int = int(os.getenv("MAX_STEPS", "10"))
TASK_ID:      str = os.getenv("TASK_ID", "anomaly_detection_easy")

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client (HF Inference Endpoints, vLLM, TGI, etc.)
# ─────────────────────────────────────────────────────────────────────────────
_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,         # Hackathon mandates HF_TOKEN as the auth key
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — instructs the LLM to behave as a compliance auditor
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a High-Frequency Trading Compliance Auditor AI.

You will receive a numbered list of flagged trades, each described by 4 float features:
  [0] time_elapsed      : Normalized age of the trade (0.0 = just arrived, 1.0+ = overdue)
  [1] price_delta       : Price discrepancy magnitude (0.0 = no mismatch detected yet)
  [2] missing_frequency : System-wide anomaly rate (0.0 = clean, 1.0 = fully anomalous)
  [3] risk_score         : Counterparty risk score (0.0 = low-risk, 1.0 = max-risk)

Decision rules:
  - decision=2 (FLAG)        : Trade is anomalous. Use when time_elapsed > 0.8, risk_score > 0.5,
                                or missing_frequency > 0.3.
  - decision=1 (INVESTIGATE) : Trade is borderline. Use when time_elapsed is 0.4–0.8 and
                                risk_score is 0.3–0.5.
  - decision=0 (PASS)        : Trade appears clean. Only use when all features are low.

You MUST respond with ONLY a valid JSON object in this exact format:
{"decisions": [<int>, <int>, ...]}

The list MUST contain exactly as many integers as there are trades provided.
Do NOT include any explanation, markdown, or extra text — only the raw JSON object.\
"""


def _ts() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_user_prompt(step: int, features: list[list[float]]) -> str:
    """Format the anomaly matrix as a numbered table for the LLM."""
    lines = [
        f"Step {step}: You have {len(features)} flagged trades to audit.",
        "",
        "Trade# | time_elapsed | price_delta | missing_freq | risk_score",
        "-------|--------------|-------------|--------------|----------",
    ]
    for i, row in enumerate(features):
        # Guard against malformed rows returned by C++ engine
        if len(row) >= 4:
            lines.append(
                f"  {i+1:3d}  |   {row[0]:8.4f}   |   {row[1]:7.4f}   |   {row[2]:8.4f}   |  {row[3]:7.4f}"
            )
        else:
            lines.append(f"  {i+1:3d}  |   (malformed row: {row})")

    lines.append("")
    lines.append(f"Provide exactly {len(features)} decisions as a JSON object.")
    return "\n".join(lines)


def _parse_llm_decisions(content: str, expected_count: int) -> list[int]:
    """
    Parse the LLM response into a list of integer decisions.

    Strategy (in priority order):
      1. Parse the full response as a JSON object and extract 'decisions'.
      2. Find the first JSON-like list in the text with regex.
      3. Fall back to a conservative all-2 (FLAG everything) default.
    """
    # Strategy 1: Full JSON object parse
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and "decisions" in parsed:
            decisions = [int(d) for d in parsed["decisions"]]
            return _normalize_decisions(decisions, expected_count)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Strategy 2: Regex extraction of a bare JSON list
    match = re.search(r'\[[\s\d,]+\]', content)
    if match:
        try:
            decisions = json.loads(match.group())
            return _normalize_decisions([int(d) for d in decisions], expected_count)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Conservative fallback — flag everything (safest for scoring)
    print(
        f"[WARN]  Could not parse LLM response. Content: {content[:200]!r}. "
        f"Defaulting to FLAG all {expected_count} trades.",
        file=sys.stderr
    )
    return [2] * expected_count


def _normalize_decisions(decisions: list[int], expected: int) -> list[int]:
    """Clamp values to [0, 2] and pad/truncate to match expected length."""
    # Clamp each decision to valid range
    clamped = [max(0, min(2, d)) for d in decisions]
    # Truncate if too long
    clamped = clamped[:expected]
    # Pad with 2 (FLAG) if too short — conservative default
    while len(clamped) < expected:
        clamped.append(2)
    return clamped


def _call_llm(step: int, features: list[list[float]]) -> list[int]:
    """Call the LLM and return parsed decisions for the given feature matrix."""
    user_prompt = _build_user_prompt(step, features)

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.0,        # deterministic for reproducible scoring
            # Note: response_format={"type": "json_object"} is NOT used here because
            # many HF-hosted endpoints do not support it. We rely on the prompt
            # instead and fall back gracefully via _parse_llm_decisions().
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN]  LLM call failed at step {step}: {e}. Defaulting to FLAG all.", file=sys.stderr)
        return [2] * len(features)

    return _parse_llm_decisions(content, len(features))


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────
def run_inference() -> None:
    episode_id: str = "unknown"
    total_reward: float = 0.0
    steps_completed: int = 0
    status: str = "SUCCESS"

    try:
        # ── Instantiate environment directly (in-process, no HTTP) ──────────
        env = FinAuditorEnvironment()
        obs = env.reset()
        episode_id = env.state.episode_id

        print(
            f"[START] episode_id={episode_id} task={TASK_ID} "
            f"max_steps={MAX_STEPS} model={MODEL_NAME} timestamp={_ts()}"
        )
        sys.stdout.flush()

        for step_num in range(1, MAX_STEPS + 1):
            features: list[list[float]] = obs.features

            # Empty matrix means no anomalies this tick — still step to advance time
            if not features:
                print(
                    f"[STEP]  step={step_num} anomalies=0 decisions=[] "
                    f"reward=0.0 cumulative_reward={total_reward:.4f} note=empty_matrix"
                )
                # Submit empty action to advance engine clock
                action = AuditorAction(decisions=[])
            else:
                # ── LLM decision-making ──────────────────────────────────────
                decisions = _call_llm(step_num, features)
                action = AuditorAction(decisions=decisions)

                print(
                    f"[STEP]  step={step_num} anomalies={len(features)} "
                    f"decisions={decisions[:10]}{'...' if len(decisions) > 10 else ''} "  # truncate for readability
                    f"reward_pending=<computed_by_env>"
                )

            # ── Execute action in environment ────────────────────────────────
            obs = env.step(action)
            step_reward: float = obs.reward if obs.reward is not None else 0.0
            total_reward += step_reward
            steps_completed = step_num

            # Re-print the STEP line with the actual reward from the environment
            print(
                f"[STEP]  step={step_num} anomalies={len(features)} "
                f"reward={step_reward:.4f} cumulative_reward={total_reward:.4f} "
                f"message={obs.message!r}"
            )
            sys.stdout.flush()

            if obs.done:
                print(f"[STEP]  step={step_num} note=episode_done_signal_received")
                break

    except KeyboardInterrupt:
        status = "INTERRUPTED"
    except Exception as exc:
        status = "ERROR"
        traceback.print_exc(file=sys.stderr)
        print(f"[ERROR] {exc}", file=sys.stderr)

    # ── Final structured log ─────────────────────────────────────────────────
    avg_reward = total_reward / max(steps_completed, 1)
    print(
        f"[END]   episode_id={episode_id} task={TASK_ID} "
        f"steps={steps_completed} total_reward={total_reward:.4f} "
        f"avg_reward={avg_reward:.4f} status={status} timestamp={_ts()}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    run_inference()
