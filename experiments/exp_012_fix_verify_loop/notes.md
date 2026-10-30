# Experiment Log: EXP-012 - Fix-Verify Loop

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: EXP-009 winner | Expected: +2 to +5 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** A `LoopAgent[fixer, critic]` that runs for up to K rounds. The critic has a fresh context (`include_contents: none`): it sees the problem and `git diff`, not the fixer's reasoning. It runs tests and writes a verdict to `review`, which the next fixer round reads.
* **Why it may work:** the baseline's main failure is submitting without verifying. A reviewer that never saw the fixer's reasoning doesn't share its blind spots.
* **How it differs from EXP-011:** this spends compute one round after another (refining one fix) instead of side by side (several attempts). Compare the two at equal wall-clock. If both help, the judge in EXP-011 can double as the critic.
* **Unknowns (EXP-005):**
  * Q6: does `{review?}` work in the first round, before `review` exists?
  * Q7: can the loop exit early? If not, a PASS round costs one short no-op turn per agent.

### Kill criteria
* Less than +2 tasks on 129 → reject.
* EXP-011 does better at equal wall-clock → keep EXP-011, or merge the two.
* The critic's false-PASS rate is above 50% → the critic adds nothing, so reject.

## 2. Configuration & Changes Made
* The root coordinator holds `submit_patch` and calls the loop as an `agent_tool`. The loop itself never submits.
* Sweep `max_iterations` over {2, 3}.

## 3. Results & Quantitative Evaluation

| Metric | Control | K=2 | K=3 |
| :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | `--` | `--` |
| **Critic false-PASS / false-FAIL** | n/a | `--` | `--` |
| **Avg rounds used** | n/a | `--` | `--` |
| **Wall-clock / task** | `--` | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] Compare head-to-head with EXP-011 at the same per-task time cap.
