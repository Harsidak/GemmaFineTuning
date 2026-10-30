# Experiment Log: EXP-006 - eval_config Pacing

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (blocked on EXP-004 for the concurrency estimate)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (repeat for E4B)
- **Primary Metric (Pass@1):** uncapped vs capped, plus projected runtime for 120 tasks

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether `eval_config.yaml` can use the 12-hour budget fully. That is the fuel-flow lever the rules openly give us.
* **Why:**
  * The default per-task cap is 60 minutes, and 120 tasks × 60 minutes is 120 hours.
  * If most tasks finish quickly that's harmless, but a few slow tasks could push the run past 12 hours, and that failure costs **every** remaining task.
  * Container setup counts toward the 12 hours but **not** toward `max_time_minutes`. Setup time therefore has to be subtracted explicitly.
* **The insurance this relies on:** at the time cap, the harness still harvests the diff in the working tree (§8.1). A cap cuts work short but keeps whatever partial work is there (see EXP-014).

### Kill criteria
* If the cap forced by the 12-hour limit costs more than 3 of 129 tasks compared with uncapped, stop and make the agent faster (EXP-015, EXP-009) instead of accepting the loss.

## 2. Configuration & Changes Made
* **Formula:** `cap = min(1.2 × p90(time of resolved tasks), 0.85 × 12h × C / 120 − mean setup time)`, where C is the concurrency from EXP-004.
* **`timeout_seconds`:** sweep 60, 120 and 300. A test suite that hangs at 300 seconds wastes a large share of a 6-minute budget.
* **`max_tool_calls`:** raise it from the `inference.py` default of 100 so that time, not tool calls, is the binding limit. The harness's `budget_warning` fires when 10 or fewer calls remain, and that warning is the agent's cue to finish up.

## 3. Results

| Metric | Uncapped | Capped | Delta |
| :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | `--` | `--` |
| **p50 / p90 resolved minutes** | `--` | `--` | |
| **Mean setup minutes** | `--` | | |
| **Projected hours for 120 tasks** | `--` | `--` | |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** the chosen `eval_config.yaml` for both submissions (31B and E4B, calibrated separately).

## 7. Actionable Next Steps
- [ ] Repeat the calibration every time the agent architecture changes. Each architecture has a different time profile.
