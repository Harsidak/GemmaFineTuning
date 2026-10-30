# Experiment Log: EXP-009 - Context-Isolated Scout

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: CTRL-31B | Expected: +2 to +6 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** The legal version of "use E4B as a scout under 31B". Mixing models is banned (Competition.md:41–55), so the scout is a **31B sub-agent made cheap**:
  * `include_contents: none`, so it never sees the root's history;
  * a small thinking budget and output limit;
  * it returns at most a 300-word localization report.
* **Why we expect gains:**
  * Every file the root reads stays in its 32k context until compaction summarises it away, and once that happens late in a session, reasoning quality drops.
  * With the scout, the root sees only the report and stays sharp for editing.
  * The harness authors recommend this pattern (§10.4).
* **Templating gotcha:** `{hints}` exists in session state only when the task has hint text (§5.1). The template must use `{hints?}`, or the agent errors on tasks without hints. EXP-005 Q6 confirms whether `?` works.

### Kill criteria
* The scout's top-3 file accuracy is below 60% → fix the scout before judging Pass@1. Accuracy is scored against the gold patch, for evaluation only.
* Pass@1 is below control + 2 → reject.
* Wall-clock per task is more than 1.3× control with no gain → reject.

## 2. Configuration & Changes Made
* **Variant A:** the scout is an `agent_tool`, so the root can call it again with a narrower question.
* **Variant B:** `SequentialAgent[scout, root]`. The scout runs once and the root receives `{localization}` in its instruction.
  * Caution (EXP-005 Q8): if the root is a pipeline, a nudge may rerun the scout. Variant A avoids this.
* **The scout** uses the code-intelligence tools plus `read_file` and `run_command` (grep and similar). Its instruction makes it read-only, and it has no `edit_file` or `submit_patch`.
* **Adapter:** in v1 the scout uses the same main adapter. A dedicated scout adapter is EXP-010.

## 3. Results & Quantitative Evaluation

| Metric | Control | Var A | Var B | Target |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | `--` | `--` | +2 tasks |
| **Scout top-1 / top-3 file acc.** | n/a | `--` | `--` | ≥ 60% top-3 |
| **Root peak prompt tokens** | `--` | `--` | `--` | ↓ |
| **Compactions / task** | `--` | `--` | `--` | ↓ |
| **Wall-clock / task** | `--` | `--` | `--` | ≤ 1.3× |

## 4. Mistakes Made & Failure Analysis
* (fill in) Watch for: the root ignores the report and re-browses anyway, or the scout writes reports that are too vague to act on.

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] If adopted, this becomes the skeleton for EXP-010 (role adapters) and EXP-011 (the scout feeding N fixers).
