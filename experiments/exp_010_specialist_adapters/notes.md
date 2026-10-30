# Experiment Log: EXP-010 - Specialist Adapters (several LoRAs, one base)

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (adapter training is owned by the Alignment session)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (also E4B)
- **Primary Metric (Pass@1):** Control: EXP-009 architecture with one shared adapter | Expected: +2 to +5 tasks | Actual: `--`

---

## 1. Hypothesis & Motivation
* **What are we testing?** Whether role-specific adapters (scout, fixer, judge) beat one shared adapter running the same multi-agent architecture.
* **The loophole:** the single-model rule restricts **base models**, not adapters (§3.4). A 3 GiB limit fits about eight r16 adapters for 31B, so we can effectively run "different models" by role.
* **Why it may work:** each role has a narrower output distribution, so a small adapter can specialise further. The judge adapter can also be trained on preference pairs taken from best-of-N rollouts (EXP-011), which fits naturally with the DPO plan.
* **Risk:** multi-LoRA batching in vLLM adds per-token overhead, which counts against our time budget.

### Kill criteria
* Less than +3 tasks over the shared adapter → reject.
* Latency with 3 LoRAs above 1.25× a single LoRA → keep only the roles that pay for themselves.

## 2. Configuration & Changes Made
* **Data:**
  * Split the existing SFT trajectories by phase: localization (start to first edit) goes to `scout_lora`, editing and verification to `fixer_lora`.
  * Judge data comes from best-of-N outcomes.
  * The 129 local tasks are never used for training.
* **Ownership:**
  * The Alignment session trains these adapters.
  * This experiment defines the data slices and runs the evaluation.
  * No changes to Alignment's trainer or configs without the user's approval.

## 3. Results & Quantitative Evaluation

| Metric | Shared adapter | Role adapters | Delta |
| :--- | :--- | :--- | :--- |
| **Pass@1 (129)** | `--` | `--` | `--` |
| **Latency ratio (multi/single LoRA)** | 1.0 | `--` | |
| **Total adapter size (MB)** | `--` | `--` | |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** / [ ] **ITERATE** / [ ] **REJECT**

## 7. Actionable Next Steps
- [ ] Hand the data-slicing spec to the Alignment session once EXP-009 has settled the architecture.
