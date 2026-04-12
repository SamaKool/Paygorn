from __future__ import annotations
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 1. IMPORT THE MEDIUM GRADER FROM THE CLASSIFICATION FILE
from graders.grader_classification import MediumClassificationGrader

TASK_ID = "anomaly_detection_medium"
MAX_STEPS = 10
DIFFICULTY = "medium"

# 2. INSTANTIATE THE MEDIUM GRADER
grader = MediumClassificationGrader()

def get_task_config() -> dict:
    return {
        "id": TASK_ID,
        "difficulty": DIFFICULTY,
        "max_steps": MAX_STEPS,
        "grader": grader,
        "description": "MEDIUM — Faster ingestion, tighter metrics."
    }

def setup_env(env) -> None:
    try:
        from server.fin_auditor_environment import hft_auditor
        if hft_auditor is not None:
            env.difficulty = hft_auditor.Difficulty.MEDIUM
            env._MAX_EPISODE_STEPS = MAX_STEPS
    except Exception as e:
        print(f"[task_medium] Could not set difficulty: {e}")


def run_episode(env, agent_fn) -> dict:
    """Run a single MEDIUM anomaly-detection episode and return the graded result.

    Called by the OpenEnv evaluator. agent_fn receives an observation and
    returns a list of binary decisions (0 = valid, 1 = anomaly).
    """
    setup_env(env)
    total_reward = 0.0
    steps_done = 0

    try:
        obs = env.reset()
        for _ in range(MAX_STEPS):
            decisions = agent_fn(obs)
            from models import AuditorAction
            action = AuditorAction(decisions=decisions)
            obs = env.step(action)
            total_reward += float(obs.reward) if obs.reward is not None else 0.0
            steps_done += 1
            if obs.done:
                break
    except Exception as exc:
        print(f"[task_medium] run_episode error at step {steps_done}: {exc}")

    # Always grade — even partial data yields a valid score via perfect_signal fallback
    final_score = grader.grade(env.state)

    return {
        "task": TASK_ID,
        "difficulty": DIFFICULTY,
        "steps": steps_done,
        "total_reward": round(total_reward, 4),
        "score": round(final_score, 4),
        "grader_breakdown": grader.last_breakdown,
    }