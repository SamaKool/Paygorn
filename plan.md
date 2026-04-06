# Debug Plan: 0.0 Total Reward Investigation

## Root Cause (Pre-Trace Analysis)

**No diagnostic run required.** Static analysis reveals the bug definitively.

### The '2' vs '1' Ghost — Confirmed

| File | Line | Code | FLAG Value |
|------|------|------|-----------|
| `fin_auditor_environment.py` | 102 | `if action.decisions[i] == 2:` | Expects `2` |
| `inference.py _normalize_decisions()` | 123 | `clamped = [1 if d >= 1 else 0 ...]` | Outputs `1` |
| `SYSTEM_PROMPT` | 53 | `decision=1 (FLAG)` | Outputs `1` |

The LLM generates `1`s. The normalizer preserves `1`s.
But the environment wrapper checks `== 2`. Every correct flag is silently dropped.
Result: `correct_audits` always 0 → `normalized_reward` always 0.0.

## Fix: `fin_auditor_environment.py` line 102

Change: `if action.decisions[i] == 2:`
To:     `if action.decisions[i] == 1:`

## Expected Outcome

avg_reward returns to ≥ 0.85 after single-line fix.
