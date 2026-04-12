from __future__ import annotations
from typing import Any

# HARD MODE: Brutal adversarial penalties.
_TP_WEIGHT = 1.0  
_TN_WEIGHT = 0.05 # Tiny reward for passing valid trades
_FP_PENALTY = 0.4 # Heavy penalty for false alarms
_FN_PENALTY = 0.8 # Massive penalty for missing adversarial trades

class HardFixGrader:
    """Grader for Task 3: Code Fix repurposed for HFT (Hard)."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, state: Any, ground_truth: dict[str, Any] | None = None) -> float:
        tp = float(getattr(state, "total_tp", 0))
        tn = float(getattr(state, "total_tn", 0))
        fp = float(getattr(state, "total_fp", 0))
        fn = float(getattr(state, "total_fn", 0))

        total = tp + tn + fp + fn
        if total == 0:
            return 0.01

        positive_signal = (tp * _TP_WEIGHT) + (tn * _TN_WEIGHT)
        negative_signal = (fp * _FP_PENALTY) + (fn * _FN_PENALTY)

        max_signal = total * _TP_WEIGHT
        raw_score = max(0.0, positive_signal - negative_signal) / max_signal

        score = max(0.01, min(0.99, raw_score))
        self.last_breakdown = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "score": score}
        return score