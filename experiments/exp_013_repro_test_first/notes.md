# Experiment Log: EXP-013 - Reproduction Test First

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (prompt-only; cheap)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: CTRL-31B | Expected: +2 to +6 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether a reproduce, then fix, then verify routine helps when the reproduction test lives **inside the repo's `tests/` folder**.
* **The loophole:**
  * The harness removes every change to `test_*.py`, `*_test.py`, `tests/**`, `conftest.py` and similar files from the patch (§8.1).
  * So the agent can write repro tests where imports and fixtures already work, and they never pollute the patch.
  * No cleanup step, no `/tmp` path problems, and no risk of a stray test file failing the scorer.
* **A related trap the prompt has to cover:**
  * The same stripping removes fixes made in `pyproject.toml`, `setup.cfg`, `tox.ini` and `pytest.ini`.
  * An agent that "fixes" a bug through config believes it succeeded, but the scorer never sees the change.
  * Also covered by `patch_check.py` in EXP-007.
* **Why we expect gains:** the baseline notes show submissions made without verification, and edits to the wrong place. A failing repro that the fix turns into a pass catches both.

### Kill criteria
* Less than +2 tasks on 129 → reject.
* More than 3 extra timeouts → require a repro only when the problem statement includes a traceback, then retest once.

## 2. Configuration & Changes Made
* A prompt change only (see `instruction_delta` in `config.yaml`). It can sit on top of any architecture.
* If adopted, the SFT trajectories should show the same routine. Coordinate with the Alignment session.

## 3. Results & Quantitative Evaluation

| Metric | Control | Target | Actual | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | +2 tasks | `--` | `--` |
| **Repro written** | `--` | > 80% | `--` | |
| **Repro failed first (real repro)** | n/a | > 60% | `--` | |
| **Repro passed but unresolved** | n/a | ↓ | `--` | |
| **Timeouts** | `--` | ≤ +3 | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in) "Repro passed but unresolved" shows where our tests are weaker than the hidden ones. Check whether those cases are about edge cases or about another call site.

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] If adopted, the critic in EXP-012 and the judge in EXP-011 can reuse each candidate's repro test as a selection signal.
