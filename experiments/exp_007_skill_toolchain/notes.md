# Experiment Log: EXP-007 - Skill Toolchain (custom tools through skills)

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (the scripts are code, so writing them needs the user's go-ahead)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: `--` | Expected: control + 3–8 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether the agent does better with deterministic Python scripts packaged as an ADK skill (`locate.py`, `related_tests.py`, `run_tests.py`, `patch_check.py`) than with raw `run_command` and `grep`.
* **The loophole:** Python entrypoints are banned and the tool registry is closed, but `.py` files **are** allowed inside skills. The agent runs them inside the task container with `run_skill_script`. In effect that's a custom toolset.
* **Why we expect gains:**
  * The baseline shows two common failures: editing the wrong place, and submitting patches that are broken or never tested. These scripts target both.
  * `patch_check.py` also catches a **silent** failure: fixes placed in `pyproject.toml`, `setup.cfg`, `tox.ini` or under `tests/` are removed from the patch by the harness (§8.1), so the agent thinks it has fixed the bug when it hasn't.
* **Possible bonus:** if EXP-005 Q2 shows that skill calls aren't counted in `max_tool_calls`, they still cost time but not tool calls.

### Kill criteria
* **DEV-30:** less than +1 task and no drop in the broken-patch rate → stop.
* **Full 129:** adopt only if the gain is at least +3 tasks with no increase in timeouts.
* If the agent ignores the scripts on more than half the tasks, revise the prompt once, then decide.

## 2. Configuration & Changes Made
* **New:**
  * `skills/swe_toolkit/SKILL.md` (frontmatter `name: swe_toolkit`)
  * `scripts/*.py`, using only the standard library and capping their own output and runtime
  * `resources/workflow.md`
* **Root agent:** add `skills: [skills/swe_toolkit]`. The prompt says to run `locate` first and `patch_check` before submitting.
* **Training interaction:** if this is adopted, the SFT trajectories must include the skill calls, or the fine-tuned adapter won't use them. Coordinate with the Alignment session.
* **Prerequisites from EXP-005:** Q2 (tool-call accounting for skill calls).

## 3. Results & Quantitative Evaluation

| Metric | Control | Target | Actual | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | +3 tasks | `--` | `--` |
| **Broken-patch rate** | `--` | ↓ | `--` | `--` |
| **Stripped-edit rate** | `--` | 0 | `--` | `--` |
| **Tool calls to first edit** | `--` | ↓ | `--` | `--` |
| **Skill usage rate** | 0 | > 80% | `--` | |
| **Timeouts** | `--` | ≤ control | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] If adopted: put skill calls into the SFT data and reuse the scripts as the verifier's tools in EXP-011 and EXP-012.
