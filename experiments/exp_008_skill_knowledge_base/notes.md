# Experiment Log: EXP-008 - Skill Knowledge Base

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (low priority, speculative)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: best after EXP-007 | Expected: +0 to +3 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether reference notes stored as skill resources help the agent. The agent loads them with `load_skill_resource` only when needed.
* **The loophole:** the rules allow up to 1,000 skills of 50 MiB each. That's a huge reference library, and unlike prompt text it uses **no context** until the agent opens a page.
* **Why it may be weak:** the hidden repos are private, so we can't write notes for them specifically. Only generic Python, pytest and popular-library knowledge applies, and the 31B model may already know most of it. E4B is more likely to benefit.
* **Hard rule:** nothing in the library may be taken from the 129 local tasks (their patches, tests or problem statements). They are our test set.

### Kill criteria
* The agent loads a resource on fewer than 10% of tasks → reject.
* The gain is less than +2 tasks on 129 → reject. Without a clear gain, the simpler submission is better.

## 2. Configuration & Changes Made
* Four skills of `.md` resources: `python_bug_patterns`, `pytest_triage`, `stdlib_api_notes` and `popular_lib_notes`.
* Each skill's `SKILL.md` description tells the agent when to load it. Nothing is loaded by default.
* Sources: public documentation and our own write-ups only.

## 3. Results & Quantitative Evaluation

| Metric | Control | Target | Actual | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | +2 tasks | `--` | `--` |
| **Resource load rate** | 0 | > 10% | `--` | |
| **Avg tokens loaded / task** | 0 | < 3k | `--` | |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] If the E4B submission benefits but 31B doesn't, ship the library only in the E4B zip.
