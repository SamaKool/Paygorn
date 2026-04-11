"""Grader for Task 2 — Conflict Classification (medium).

Scoring
-------
    1.0  — exact match with the ground-truth violation type
    0.5  — same constraint family (resource-limit or temporal-ordering)
    0.1  — valid category but from a different family
    0.0  — empty or completely unrecognised response

Constraint families (related groups for partial credit)
-------------------------------------------------------
    Resource-limit family : resource_overload, capacity_exceeded
        Both concern the number of jobs concurrently on a machine.
    Temporal-ordering family : deadline_violation, precedence_violation
        Both concern the sequencing and timing of job execution.
    Standalone : availability_conflict
        Concerns machine operational windows (no close sibling).
        
After each call, ``last_breakdown`` holds a dict describing the decision.
"""

from __future__ import annotations

from typing import Any

from models import Action

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "resource_overload",
        "deadline_violation",
        "precedence_violation",
        "availability_conflict",
        "capacity_exceeded",
    }
)

# Groups of semantically related categories; membership earns partial credit.
_RELATED_GROUPS: list[frozenset[str]] = [
    frozenset({"resource_overload", "capacity_exceeded"}),    # resource-limit family
    frozenset({"deadline_violation", "precedence_violation"}), # temporal-ordering family
]


def _same_family(a: str, b: str) -> bool:
    """Return True if a and b belong to the same related group."""
    return any(a in g and b in g for g in _RELATED_GROUPS)


class ConflictGrader:
    """Grade the agent's constraint-violation classification."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        # Normalise to snake_case (agents often write "deadline violation" etc.)
        response: str = (
            action.response.strip().lower().replace(" ", "_").replace("-", "_")
        )
        expected: str = ground_truth.get("violation_type") or ""

        if not response:
            self._record("", expected, 0.0, "Empty response.")
            return 0.0

        # Exact match
        if response == expected:
            self._record(response, expected, 1.0, "Exact match.")
            return 1.0

        # Not in vocabulary
        if response not in VALID_CATEGORIES:
            self._record(
                response, expected, 0.0,
                f"'{response}' is not a valid category. "
                f"Choose from: {', '.join(sorted(VALID_CATEGORIES))}.",
            )
            return 0.0

        # Same constraint family → partial credit
        if _same_family(response, expected):
            self._record(
                response, expected, 0.5,
                f"Related category (same family as '{expected}').",
            )
            return 0.5

        # Valid but different family
        self._record(
            response, expected, 0.1,
            f"Valid category but wrong family. Expected '{expected}'.",
        )
        return 0.1

    def _record(
        self, predicted: str, expected: str, score: float, feedback: str
    ) -> None:
        self.last_breakdown = {
            "predicted": predicted,
            "expected": expected,
            "score": score,
            "in_valid_categories": predicted in VALID_CATEGORIES,
            "same_family": _same_family(predicted, expected) if predicted and expected else False,
            "exact_match": predicted == expected,
            "feedback": feedback,
        }
