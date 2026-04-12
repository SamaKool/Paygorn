from __future__ import annotations
from typing import Any

# EASY MODE: Forgiving penalties.
_TP_WEIGHT = 1.0  
_TN_WEIGHT = 0.1  
_FP_PENALTY = 0.1 
_FN_PENALTY = 0.2 

class EasyDetectionGrader:
    """Grader for Task 1: Anomaly Detection (Easy)."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, state: Any = None, ground_truth: dict[str, Any] | None = None) -> float:
        if state is None:
            self.last_breakdown = {"error": "empty_state_ping", "score": 0.1}
            return 0.1

        tp = float(getattr(state, "total_tp", 0))
        tn = float(getattr(state, "total_tn", 0))
        fp = float(getattr(state, "total_fp", 0))
        fn = float(getattr(state, "total_fn", 0))

        actual_anomalies = tp + fn
        actual_valid = tn + fp
        perfect_signal = (actual_anomalies * _TP_WEIGHT) + (actual_valid * _TN_WEIGHT)

        if perfect_signal == 0:
            return 0.1

        positive_signal = (tp * _TP_WEIGHT) + (tn * _TN_WEIGHT)
        negative_signal = (fp * _FP_PENALTY) + (fn * _FN_PENALTY)

        raw_score = max(0.0, positive_signal - negative_signal) / perfect_signal
        score = max(0.1, min(0.99, raw_score))
        self.last_breakdown = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "score": score}
        return score