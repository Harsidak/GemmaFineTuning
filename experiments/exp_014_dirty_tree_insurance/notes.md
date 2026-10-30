# Experiment Log: EXP-014 - Dirty-Tree Insurance

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (prompt-only; cheap)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: CTRL-31B under the EXP-006 cap | Expected: +1 to +4 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether a policy of always keeping the best fix so far applied in `/workspace` (and always importable) wins tasks that currently end empty-handed.
* **The loophole:**
  * When an agent times out, runs out of tool calls or runs out of nudges, the harness **still** takes the working-tree diff and scores it (§8.1).
  * Phase 2 also applies it with 4 increasingly forgiving passes (§8.2), so formatting problems rarely cost us.
  * An agent that stops in the middle of a task can still score, as long as its tree holds a real fix.
* **Why it matters more under tight caps:** if EXP-004 shows the scorer runs tasks one at a time, the per-task cap is about 5–6 minutes, and many tasks will end on the cap instead of with `submit_patch`.
* **Risk:** editing early may anchor the model on a wrong first idea.

### Kill criteria
* Pass@1 drops by more than 2 tasks → drop the "edit early" rule and keep the other three: never reset the tree, run `py_compile` after every edit, and finish up on `budget_warning`.

## 2. Configuration & Changes Made
* A prompt policy only (`instruction_delta` in `config.yaml`).
* **Time awareness:** the agent can't see a clock; only `budget_warning` (tool calls) reaches it. Option: calibrate `max_tool_calls` in EXP-006 so the warning arrives at about 80% of the time cap.

## 3. Results & Quantitative Evaluation

| Metric | Control | Target | Actual | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | +1 task | `--` | `--` |
| **Empty patch on unfinished tasks** | `--` | ↓ | `--` | `--` |
| **Resolved without submit_patch** | `--` | ↑ | `--` | `--` |
| **Syntax-broken tree at end** | `--` | 0 | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] If adopted, include this policy in every later architecture's prompts, and in the SFT trajectories.
