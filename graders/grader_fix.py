"""Grader for Task 3 — Schedule Repair (hard).

Scoring breakdown (additive, max 1.0)
--------------------------------------
    0.20  — response is parseable JSON
    0.20  — JSON has the required schema (assignments list, all jobs covered)
    0.40  — schedule satisfies all constraints (0.10 per category):
              capacity, deadlines, precedence, availability
    0.20  — makespan within 30% of optimal (0.10 partial if within 60%)
        
Partial-progress signal
-----------------------
Even a structurally invalid JSON attempt earns 0.0 (wrong format).
A parseable but schema-invalid JSON earns 0.20 (gave a JSON object).
A valid schema with partial constraint satisfaction earns up to 0.80.
This dense reward curve supports multi-step improvement within an episode.

After each call, ``last_breakdown`` holds a full dict with per-category
pass/fail flags, makespan, and the optimality ratio — surfaced in the
environment's info dict.
"""

from __future__ import annotations

import json
import re
from typing import Any

from models import Action


class RepairGrader:
    """Grade the agent's proposed schedule repair."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response: str = action.response.strip()
        instance: dict[str, Any] = ground_truth.get("instance", {})
        optimal_makespan: int = int(ground_truth.get("optimal_makespan", 1) or 1)

        if not response:
            self._record_breakdown(
                json_ok=False, schema_ok=False,
                constraint_detail={}, makespan=0,
                optimal_makespan=optimal_makespan,
            )
            return 0.0

        score = 0.0

        # ------------------------------------------------------------------
        # Component 1a — Is the response parseable JSON? (0.20)
        # ------------------------------------------------------------------
        parsed = self._parse_json(response)
        if parsed is None:
            self._record_breakdown(
                json_ok=False, schema_ok=False,
                constraint_detail={}, makespan=0,
                optimal_makespan=optimal_makespan,
            )
            return 0.0  # not JSON → no partial credit at all

        score += 0.20  # JSON parseable

        # ------------------------------------------------------------------
        # Component 1b — Does it have the required schema? (0.20)
        # Required: {"assignments": [{"job_id", "machine_id", "start_time"}, ...]}
        # All jobs from the instance must be present exactly once.
        # ------------------------------------------------------------------
        assignments: list[Any] = parsed.get("assignments", [])
        schema_ok = self._valid_schema(assignments, instance)
        if not schema_ok:
            self._record_breakdown(
                json_ok=True, schema_ok=False,
                constraint_detail={}, makespan=0,
                optimal_makespan=optimal_makespan,
            )
            return round(score, 4)  # only 0.20

        score += 0.20  # valid schema

        # ------------------------------------------------------------------
        # Component 2 — Constraint satisfaction (0.40, 0.10 per category)
        # Categories: capacity, deadlines, precedence, availability
        # ------------------------------------------------------------------
        constraint_detail = self._check_constraints_detail(assignments, instance)
        satisfied = sum(constraint_detail.values())
        score += 0.40 * (satisfied / max(len(constraint_detail), 1))

        # ------------------------------------------------------------------
        # Component 3 — Makespan optimality (0.20)
        # Full 0.20 if makespan ≤ optimal × 1.30; partial 0.10 if ≤ 1.60.
        # ------------------------------------------------------------------
        makespan = self._compute_makespan(assignments, instance)
        if makespan > 0 and optimal_makespan > 0:
            ratio = makespan / optimal_makespan
            if ratio <= 1.30:
                score += 0.20
            elif ratio <= 1.60:
                score += 0.10  # partial optimality credit

        self._record_breakdown(
            json_ok=True, schema_ok=True,
            constraint_detail=constraint_detail,
            makespan=makespan,
            optimal_makespan=optimal_makespan,
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Breakdown recording
    # ------------------------------------------------------------------

    def _record_breakdown(
        self,
        json_ok: bool,
        schema_ok: bool,
        constraint_detail: dict[str, bool],
        makespan: int,
        optimal_makespan: int,
    ) -> None:
        ratio = (
            round(makespan / optimal_makespan, 3)
            if (makespan > 0 and optimal_makespan > 0)
            else None
        )
        self.last_breakdown = {
            "json_parseable": json_ok,
            "schema_valid": schema_ok,
            "constraints": constraint_detail,
            "constraints_satisfied": sum(constraint_detail.values()) if constraint_detail else 0,
            "makespan": makespan,
            "optimal_makespan": optimal_makespan,
            "makespan_ratio": ratio,
            "within_30pct": ratio is not None and ratio <= 1.30,
        }

    # ------------------------------------------------------------------
    # JSON parsing — robust to markdown fences and partial wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(response: str) -> dict[str, Any] | None:
        """Try multiple strategies to extract a JSON object from the response.

        Strategy 1: Direct json.loads (agent returned pure JSON).
        Strategy 2: Strip markdown code fences, then parse.
        Strategy 3: Brace-counting to find the outermost {...} block.
                    This is the most robust and handles agents that wrap JSON
                    in prose like "Here is my answer: {...}".
        """
        # Strategy 1 — direct parse
        try:
            obj = json.loads(response)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2 — strip code fences
        stripped = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3 — brace-counting for the outermost { ... }
        start = response.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(response[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = response[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, ValueError):
                        return None
        return None

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_schema(
        assignments: list[Any], instance: dict[str, Any]
    ) -> bool:
        """Validate that assignments is a well-formed list covering all jobs."""
        if not isinstance(assignments, list) or len(assignments) == 0:
            return False

        required_keys = {"job_id", "machine_id", "start_time"}
        for a in assignments:
            if not isinstance(a, dict):
                return False
            if not required_keys.issubset(a.keys()):
                return False
            if not isinstance(a.get("start_time"), (int, float)):
                return False
            if a.get("start_time") < 0:
                return False  # negative start times are never valid

        # Every job in the instance must appear exactly once
        expected_jobs = {j["id"] for j in instance.get("jobs", [])}
        assigned_jobs = [a["job_id"] for a in assignments]
        return set(assigned_jobs) == expected_jobs and len(assigned_jobs) == len(expected_jobs)

    # ------------------------------------------------------------------
    # Constraint checking (returns per-category bool dict)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_constraints_detail(
        assignments: list[dict[str, Any]], instance: dict[str, Any]
    ) -> dict[str, bool]:
        """Return a dict of {constraint_name: passed} for each of the 4 categories."""
        jobs_by_id = {j["id"]: j for j in instance.get("jobs", [])}
        machines_by_id = {m["id"]: m for m in instance.get("machines", [])}
        assign_by_job = {a["job_id"]: a for a in assignments}

        # ---- (a) Capacity: concurrent jobs on any machine ≤ its capacity ----
        machine_intervals: dict[str, list[tuple[float, float]]] = {}
        for a in assignments:
            mid = a["machine_id"]
            st = float(a["start_time"])
            dur = float(jobs_by_id.get(a["job_id"], {}).get("duration", 1))
            machine_intervals.setdefault(mid, []).append((st, st + dur))

        capacity_ok = True
        for mid, intervals in machine_intervals.items():
            cap = machines_by_id.get(mid, {}).get("capacity", 1)
            for s1, e1 in intervals:
                # Count how many intervals overlap with [s1, e1)
                concurrent = sum(
                    1 for s2, e2 in intervals if s2 < e1 and e2 > s1
                )
                if concurrent > cap:
                    capacity_ok = False
                    break
            if not capacity_ok:
                break

        # ---- (b) Deadlines: every job finishes by its deadline ----
        deadline_ok = True
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            finish = float(a["start_time"]) + float(job.get("duration", 0))
            dl = job.get("deadline", float("inf"))
            if finish > dl:
                deadline_ok = False
                break

        # ---- (c) Precedence: job starts after ALL its predecessors finish ----
        precedence_ok = True
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            for dep_id in job.get("dependencies", []):
                dep_a = assign_by_job.get(dep_id)
                if dep_a is None:
                    precedence_ok = False
                    break
                dep_job = jobs_by_id.get(dep_id, {})
                dep_finish = float(dep_a["start_time"]) + float(
                    dep_job.get("duration", 0)
                )
                if float(a["start_time"]) < dep_finish:
                    precedence_ok = False
                    break
            if not precedence_ok:
                break

        # ---- (d) Availability: job runs within machine availability window ----
        availability_ok = True
        for a in assignments:
            machine = machines_by_id.get(a["machine_id"], {})
            avail_start = float(machine.get("available_start", 0))
            avail_end = float(machine.get("available_end", float("inf")))
            job = jobs_by_id.get(a["job_id"], {})
            job_start = float(a["start_time"])
            job_end = job_start + float(job.get("duration", 0))
            if job_start < avail_start or job_end > avail_end:
                availability_ok = False
                break

        return {
            "capacity": capacity_ok,
            "deadlines": deadline_ok,
            "precedence": precedence_ok,
            "availability": availability_ok,
        }

    @staticmethod
    def _check_constraints(
        assignments: list[dict[str, Any]], instance: dict[str, Any]
    ) -> float:
        """Convenience wrapper — returns fraction of categories satisfied."""
        detail = RepairGrader._check_constraints_detail(assignments, instance)
        return sum(detail.values()) / max(len(detail), 1)

    # ------------------------------------------------------------------
    # Makespan calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_makespan(
        assignments: list[dict[str, Any]], instance: dict[str, Any]
    ) -> int:
        """Return the latest finish time across all assigned jobs."""
        jobs_by_id = {j["id"]: j for j in instance.get("jobs", [])}
        max_finish = 0
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            finish = int(a["start_time"]) + int(job.get("duration", 0))
            if finish > max_finish:
                max_finish = finish
        return max_finish
