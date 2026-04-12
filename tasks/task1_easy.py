from __future__ import annotations
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 1. IMPORT THE EASY GRADER FROM THE DETECTION FILE
from graders.grader_detection import EasyDetectionGrader

TASK_ID = "anomaly_detection_easy"
MAX_STEPS = 5
DIFFICULTY = "easy"

# 2. INSTANTIATE THE EASY GRADER
grader = EasyDetectionGrader()

def get_task_config() -> dict:  
    return {
        "id": TASK_ID,
        "difficulty": DIFFICULTY,
        "max_steps": MAX_STEPS,
        "grader": grader,
        "description": "EASY — Detect expired/unreconciled trades."
    }

def setup_env(env) -> None:
    try:
        from server.fin_auditor_environment import hft_auditor
        if hft_auditor is not None:
            env.difficulty = hft_auditor.Difficulty.EASY
            env._MAX_EPISODE_STEPS = MAX_STEPS
    except Exception as e:
        print(f"[task_easy] Could not set difficulty: {e}")