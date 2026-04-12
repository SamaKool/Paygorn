from __future__ import annotations
import math
from typing import Any

# HARD MODE: Brutal adversarial penalties.
_TP_WEIGHT = 1.0
_TN_WEIGHT = 0.05  # Tiny reward for passing valid trades
_FP_PENALTY = 0.4  # Heavy penalty for false alarms
_FN_PENALTY = 0.8  # Massive penalty for missing adversarial trades

# Hard score boundaries required by the competition evaluator.
# Scores must be strictly between 0 and 1 (exclusive).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

class HardFixGrader:
    """Grader for Task 3: Code Fix repurposed for HFT (Hard).
    
    Score is always in the open interval (0, 1) — never 0.0, never 1.0.
    """

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, env: Any = None, *args, **kwargs) -> float:
        """Return a float strictly in (_SCORE_MIN, _SCORE_MAX)."""
        if env is None:
            self.last_breakdown = {"error": "empty_env_ping", "score": _SCORE_MIN}
            return _SCORE_MIN

        state = getattr(env, "_state", env)
        tp = float(getattr(state, "total_tp", 0))
        tn = float(getattr(state, "total_tn", 0))
        fp = float(getattr(state, "total_fp", 0))
        fn = float(getattr(state, "total_fn", 0))

        actual_anomalies = tp + fn
        actual_valid = tn + fp
        perfect_signal = (actual_anomalies * _TP_WEIGHT) + (actual_valid * _TN_WEIGHT)

        if perfect_signal == 0:
            self.last_breakdown = {"error": "zero_perfect_signal", "score": _SCORE_MIN}
            return _SCORE_MIN

        positive_signal = (tp * _TP_WEIGHT) + (tn * _TN_WEIGHT)
        negative_signal = (fp * _FP_PENALTY) + (fn * _FN_PENALTY)

        raw_score = max(0.0, positive_signal - negative_signal) / perfect_signal
        
        # Guard against zero-division NaN or infinity escaping
        if not math.isfinite(raw_score):
            raw_score = 0.0

         # Double-clamp: first to [0,1] then to our exclusive-boundary window.
        score = float(max(_SCORE_MIN, min(_SCORE_MAX, raw_score)))
        self.last_breakdown = {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "score": round(score, 4),
        }
        return score
