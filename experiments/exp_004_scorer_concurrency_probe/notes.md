# Experiment Log: EXP-004 - Scorer Concurrency Probe

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (**uses a Kaggle submission, so it needs the user's go-ahead**)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-e4b-it-qat-w4a16-ct` (probe only)
- **Primary Metric:** whether the submission finishes within the 12-hour limit (Pass@1 is irrelevant)

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether the Kaggle scorer runs the ~120 hidden tasks one at a time or several at once.
* **Why it matters:** this sets the per-task time budget, and every other experiment depends on that number.
  If tasks run one at a time, we get about 5–6 minutes per task and parallel tricks inside a task (EXP-011, EXP-012) are worth more. If they run in parallel, the budget per task is much larger.
* **What we know:**
  * `eval_config.yaml` has no concurrency setting.
  * The local CLI supports `--concurrency N`, but the Kaggle `scripts/inference.py` isn't published.
  * The default `max_time_minutes` is 60, and 60 minutes × 120 tasks far exceeds 12 hours, which suggests either short tasks or tasks running in parallel.
* **Legality:** this is a normal submission. It just spends its time sleeping.

## 2. Configuration & Changes Made
* **Probe P1:**
  * The agent calls `sleep 290` twice and then `submit_patch()`.
  * That's about 10 minutes per task plus setup, so roughly 20 hours if tasks run one at a time.
  * If P1 finishes within 12 hours, concurrency is at least 2.
* **Probe P2** (only if P1 passes):
  * Six sleeps, about 30 minutes per task, so roughly 60 hours if run one at a time.
  * If P2 finishes, concurrency is at least about 5–6.
* Use E4B with `thinking_budget: 0` and temperature 0, so the model's own run time barely affects the measurement.
* Before submitting, check locally that the probe agent really makes the sleep calls. Run 2 tasks with `swegemma eval` and read the trace.

## 3. Results

| Probe | Serial-equivalent runtime | Completed within 12h? | Concurrency lower bound |
| :--- | :--- | :--- | :--- |
| P1 | ~20h | `--` | `--` |
| P2 | ~60h | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in) Did the agent skip the sleeps or call `submit_patch` early? If so, the probe proves nothing.

## 5. Root Cause Analysis (RCA)
* n/a (measurement)

## 6. Final Verdict & Decision
- [ ] Concurrency is 1: budget about 5–6 minutes per task and prioritise parallel work inside a task.
- [ ] Concurrency is 2 or more: recompute the per-task budget in EXP-006.

## 7. Actionable Next Steps
- [ ] Before using a submission, ask on the competition discussion forum (free).
- [ ] Next experiment: `EXP-006`
