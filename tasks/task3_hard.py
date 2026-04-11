from __future__ import annotations
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graders.grader_detection import FinAuditorGrader

TASK_ID = "anomaly_detection_hard"
MAX_STEPS = 20
DIFFICULTY = "hard"

grader = FinAuditorGrader()

def get_task_config() -> dict:
    return {
        "id": TASK_ID,
        "difficulty": DIFFICULTY,
        "max_steps": MAX_STEPS,
        "grader": grader,
        "description": "HARD — Maximum throughput adversarial trading."
    }

def setup_env(env) -> None:
    try:
        from server.fin_auditor_environment import hft_auditor
        if hft_auditor is not None:
            env.difficulty = hft_auditor.Difficulty.HARD
            env._MAX_EPISODE_STEPS = MAX_STEPS
    except Exception as e:
        print(f"[task_hard] Could not set difficulty: {e}")