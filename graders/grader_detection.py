"""Graders for detection-class tasks.

FeasibilityGrader
-----------------
    Grades Task 1 — binary feasible / infeasible schedule check.
    Scores: 1.0 exact match | 0.1 wrong answer | 0.0 empty.

FinAuditorGrader
----------------
    Grades HFT Auditor episodes using C++ ReconciliationEngine metrics.
    Called by OpenEnv automatically when done=True.
    Score formula: asymmetric TP/FP/FN weighting, clamped strictly to [0.01, 0.99].
"""

from __future__ import annotations

from typing import Any

from models import Action

# Words treated as equivalent to "feasible"
_FEASIBLE_WORDS: frozenset[str] = frozenset(
    {"feasible", "valid", "correct", "satisfiable", "yes", "ok", "pass"}
)

# Words treated as equivalent to "infeasible"
_INFEASIBLE_WORDS: frozenset[str] = frozenset(
    {
        "infeasible", "invalid", "incorrect", "unsatisfiable", "no",
        "violated", "conflict", "fail", "impossible", "broken",
    }
)


class FeasibilityGrader:
    """Grade whether the agent correctly determined schedule feasibility."""

    def __init__(self) -> None:
        # Populated after each call to grade(); surfaced in env info dict.
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response: str = action.response.strip().lower()
        is_feasible: bool = ground_truth.get("is_feasible", False)
        expected: str = "feasible" if is_feasible else "infeasible"

        # Empty response → no signal
        if not response:
            self.last_breakdown = {
                "predicted": "",
                "expected": expected,
                "correct": False,
                "feedback": "Empty response — reply with 'feasible' or 'infeasible'.",
            }
            return 0.0

        # Normalise response to canonical form
        if response in _FEASIBLE_WORDS:
            predicted = "feasible"
        elif response in _INFEASIBLE_WORDS:
            predicted = "infeasible"
        else:
            # Recognisable attempt but could not be parsed cleanly
            self.last_breakdown = {
                "predicted": response,
                "expected": expected,
                "correct": False,
                "feedback": (
                    f"Could not parse '{response}'. "
                    "Use exactly 'feasible' or 'infeasible'."
                ),
            }
            return 0.1

        correct = predicted == expected
        self.last_breakdown = {
            "predicted": predicted,
            "expected": expected,
            "correct": correct,
            "feedback": (
                "Correct."
                if correct
                else f"Wrong — the schedule is {expected}, not {predicted}."
            ),
        }
        # Exact match → 1.0; wrong normalised answer → 0.1 (keeps gradient signal)
        return 1.0 if correct else 0.1


# ── HFT Auditor Grader ────────────────────────────────────────────────────────

# Asymmetric reward weights matching the C++ ReconciliationEngine constants
_TP_WEIGHT: float = 1.0   # correctly flagged anomaly — full credit
_TN_WEIGHT: float = 0.1   # correctly passed valid trade — small positive
_FP_PENALTY: float = 0.1  # flagged a valid trade — minor penalty
_FN_PENALTY: float = 0.4  # missed an anomaly — severe penalty


class FinAuditorGrader:
    """Grade a completed HFT audit episode from C++ engine metrics.

    Called by OpenEnv automatically when ``done=True`` is returned by
    ``FinAuditorEnvironment.step()``.

    The score is computed from the cumulative confusion-matrix counters
    accumulated across the full episode by the C++ ReconciliationEngine:

        last_tp — True Positives  (anomalous trade correctly flagged)
        last_tn — True Negatives  (valid trade correctly passed)
        last_fp — False Positives (valid trade wrongly flagged)
        last_fn — False Negatives (anomalous trade missed — catastrophic)

    Hackathon rule: final score is strictly clamped to [0.01, 0.99].
    """

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, state: Any, ground_truth: dict[str, Any] | None = None) -> float:
        """Compute the final episode score.

        Args:
            state:        Environment state object at episode end.
                          Must expose last_tp, last_tn, last_fp, last_fn.
            ground_truth: Unused — ground truth is implicit in the C++ engine.

        Returns:
            float strictly in (0.01, 0.99).
        """
        tp = float(getattr(state, "last_tp", 0))
        tn = float(getattr(state, "last_tn", 0))
        fp = float(getattr(state, "last_fp", 0))
        fn = float(getattr(state, "last_fn", 0))

        total = tp + tn + fp + fn
        if total == 0:
            self._record(tp, tn, fp, fn, 0.01, "No trades evaluated — floor score.")
            return 0.01

        positive_signal = (tp * _TP_WEIGHT) + (tn * _TN_WEIGHT)
        negative_signal = (fp * _FP_PENALTY) + (fn * _FN_PENALTY)

        # Normalise against the theoretical maximum (all trades are TP)
        max_signal = total * _TP_WEIGHT
        raw_score = max(0.0, positive_signal - negative_signal) / max_signal

        # Strict hackathon boundary — must not be exactly 0.0 or 1.0
        score = max(0.01, min(0.99, raw_score))

        self._record(
            tp, tn, fp, fn, score,
            f"tp={int(tp)} tn={int(tn)} fp={int(fp)} fn={int(fn)} | raw={raw_score:.4f}"
        )
        return score

    def _record(
        self,
        tp: float, tn: float, fp: float, fn: float,
        score: float, feedback: str,
    ) -> None:
        total = tp + tn + fp + fn
        self.last_breakdown = {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "total": int(total),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall":    round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "score": round(score, 4),
            "feedback": feedback,
        }
        print(f"[GRADER] Episode scored: {feedback} => {score:.4f}")
