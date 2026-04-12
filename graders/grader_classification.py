from __future__ import annotations
from typing import Any

# MEDIUM MODE: Standard HFT penalties.
_TP_WEIGHT = 1.0  
_TN_WEIGHT = 0.1  
_FP_PENALTY = 0.2 # Stricter false positive
_FN_PENALTY = 0.4 # Standard catastrophic failure penalty

class MediumClassificationGrader:
    """Grader for Task 2: Conflict Classification repurposed for HFT (Medium)."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, state: Any, ground_truth: dict[str, Any] | None = None) -> float:
        tp = float(getattr(state, "total_tp", 0))
        tn = float(getattr(state, "total_tn", 0))
        fp = float(getattr(state, "total_fp", 0))
        fn = float(getattr(state, "total_fn", 0))

        total = tp + tn + fp + fn
        
        # The maximum possible score if they made zero mistakes
        actual_anomalies = tp + fn
        actual_valid = tn + fp
        
        perfect_signal = (actual_anomalies * _TP_WEIGHT) + (actual_valid * _TN_WEIGHT)
        
        if perfect_signal == 0:
            return 0.1

        positive_signal = (tp * _TP_WEIGHT) + (tn * _TN_WEIGHT)
        negative_signal = (fp * _FP_PENALTY) + (fn * _FN_PENALTY)

        # Normalize against the true perfect scenario
        raw_score = max(0.0, positive_signal - negative_signal) / perfect_signal

        # Strict hackathon boundary
        score = max(0.1, min(0.99, raw_score))
        
        self.last_breakdown = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "score": score}
        return score