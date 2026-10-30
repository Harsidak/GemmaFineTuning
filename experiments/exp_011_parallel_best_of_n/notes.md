# Experiment Log: EXP-011 - Parallel Best-of-N in Worktrees

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B, possibly with a larger N)
- **Primary Metric (Pass@1):** Control: EXP-009 winner | Expected: +4 to +10 tasks at N=3 | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether running N fixers at the same time, each in its own `/tmp` git worktree, with a judge that tests the candidates and applies the best one to `/workspace`, beats a single fixer.
* **The loophole (the "fuel-flow" one):**
  * The scorer measures time per task, not GPU work per task.
  * All agents share one vLLM server, and on L4s text generation is limited by memory bandwidth, so batching 3 requests costs only a little more time than 1.
  * `ParallelAgent` gives us those concurrent requests, so N attempts cost about the time of one.
* **Why we expect gains:** on SWE tasks, pass@k is usually well above pass@1. Even a judge that only checks "tests pass" captures part of that gap.
* **The traps this design avoids:**
  1. **Shared `/workspace`:** fixers work only in their own worktree, and only the judge writes to `/workspace`.
  2. **`submit_patch` ends the session** (EXP-005 Q5): only the judge has it.
  3. **The editable install points to `/workspace`:** tests in a worktree must set `PYTHONPATH` to that worktree, or they test the wrong code (EXP-005 Q4).
  4. **4 GiB RAM and 2 vCPUs:** fixers run only targeted tests, and the judge runs full checks one at a time.
  5. **Nudge reruns:** the root stays a plain `LlmAgent`, so a nudge doesn't relaunch the whole fan-out (EXP-005 Q8).

### Kill criteria
* N=2 gives less than control + 3 tasks on 129 → reject this line.
* Wall-clock per task at the chosen N exceeds the EXP-006 cap → reduce N or reject.
* Out-of-memory kills on more than 5% of tasks → redesign how tests run first.

## 2. Configuration & Changes Made
* **Agents:** root coordinator → scout (EXP-009) → fan-out `ParallelAgent[fixer_1..N]` → judge.
* **Diversity:** each fixer gets its own seed, a temperature from {0.0, 0.4, 0.7}, and optionally a different strategy hint.
* **Also measure the oracle score** (any candidate passes). It separates "the fixers are too weak" from "the judge picks badly".
* **Prerequisites:**
  * EXP-004: the time budget.
  * EXP-005 Q1, Q3, Q4, Q5 and Q8.
  * EXP-009: the scout.

## 3. Results & Quantitative Evaluation

| N | Pass@1 (129) | Oracle (any passes) | Judge accuracy | Wall-clock / task | OOM rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `--` | `--` | n/a | `--` | `--` |
| 2 | `--` | `--` | `--` | `--` | `--` |
| 3 | `--` | `--` | `--` | `--` | `--` |
| 4 | `--` | `--` | `--` | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in) Watch for: fixers writing to `/workspace`, the judge applying a diff that fails `git apply`, or all candidates being identical (too little diversity).

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] Rollouts from this experiment (passing vs failing candidates on the same task) make natural DPO pairs. Share them with the Alignment session, using training-pool tasks only and never the 129.
