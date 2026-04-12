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