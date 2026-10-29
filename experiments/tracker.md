# Experiments Master Tracker & Leaderboard

Use this file to monitor all experiment runs, key metric changes, and decisions across the project lifecycle.

## Leaderboard Summary

| Exp ID | Date | Name | Model Variant | Changes / Hypothesis | Val Pass@1 | Status | Verdict | Submission Packaged |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-000** | 2026-10-16 | Baseline Zero-Shot | `gemma-4-12b-it` | Standard ADK prompt + zero-shot inference | `14.2%` | COMPLETED | BASELINE | [x] |
| **EXP-001** | Planned | LoRA SFT on Trajectories | `gemma-4-12b-it` | SFT on 5k filtered SWE-bench trajectories | `--` | PLANNED | PENDING | [ ] |
| **EXP-002** | Planned | Multi-Agent Decomposition | `gemma-4-12b-it` | Split into Localizer, Patcher, and Verifier | `--` | PLANNED | PENDING | [ ] |
| **EXP-003** | Planned | 31B QAT w4a16 Scale | `gemma-4-31b-it-qat` | Upgrade to 31B QAT model with best LoRA | `--` | PLANNED | PENDING | [ ] |

---

## Experiment Index

* [EXP-000: Baseline Zero-Shot Evaluation](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/exp_000_baseline/notes.md)
* *Template:* [experiments/template.md](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/template.md)
* *Schema:* [experiments/template.yaml](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/template.yaml)
