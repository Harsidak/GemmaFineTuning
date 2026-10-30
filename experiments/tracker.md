# Experiments Master Tracker & Leaderboard

Use this file to monitor all experiment runs, key metric changes, and decisions across the project lifecycle.

## Leaderboard Summary

| Exp ID | Date | Name | Model Variant | Changes / Hypothesis | Val Pass@1 | Status | Verdict | Submission Packaged |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-000** | 2026-10-16 | Baseline Zero-Shot | `gemma-4-12b-it` | Standard ADK prompt + zero-shot inference | `14.2%` | COMPLETED | BASELINE | [x] |
| **EXP-001** | Planned | LoRA SFT on Trajectories | `gemma-4-12b-it` | SFT on 5k filtered SWE-bench trajectories | `--` | PLANNED | PENDING | [ ] |
| **EXP-002** | Planned | Multi-Agent Decomposition | `gemma-4-12b-it` | Split into Localizer, Patcher, and Verifier | `--` | PLANNED | PENDING | [ ] |
| **EXP-003** | Planned | 31B QAT w4a16 Scale | `gemma-4-31b-it-qat` | Upgrade to 31B QAT model with best LoRA | `--` | PLANNED | PENDING | [ ] |
| **EXP-004** | Planned | Scorer Concurrency Probe | `gemma-4-e4b-it-qat` | Slow "sleep" agent shows whether Kaggle runs tasks in parallel (**uses 1 submission**) | n/a | PLANNED | PENDING | [ ] |
| **EXP-005** | Planned | Harness Mechanics Spike | `gemma-4-e4b-it-qat` | Answer Q1–Q8: budget accounting, ParallelAgent concurrency, worktrees, loop exit | n/a | PLANNED | PENDING | [ ] |
| **EXP-006** | Planned | eval_config Pacing | `gemma-4-31b-it-qat` (+E4B) | Set per-task caps to fill 12h without the run being killed | `--` | PLANNED | PENDING | [ ] |
| **EXP-007** | Planned | Skill Toolchain | `gemma-4-31b-it-qat` (+E4B) | Stdlib `.py` scripts in a skill = custom tools (locate, tests, patch check) | `--` | PLANNED | PENDING | [ ] |
| **EXP-008** | Planned | Skill Knowledge Base | `gemma-4-31b-it-qat` (+E4B) | On-demand reference notes as skill resources (no context cost until loaded) | `--` | PLANNED | PENDING | [ ] |
| **EXP-009** | Planned | Context-Isolated Scout | `gemma-4-31b-it-qat` (+E4B) | Cheap 31B scout sub-agent (`include_contents: none`) returns a ≤300-word localization report | `--` | PLANNED | PENDING | [ ] |
| **EXP-010** | Planned | Specialist Adapters | `gemma-4-31b-it-qat` (+E4B) | Role LoRAs (scout/fixer/judge) on one base vs one shared LoRA | `--` | PLANNED | PENDING | [ ] |
| **EXP-011** | Planned | Parallel Best-of-N | `gemma-4-31b-it-qat` (+E4B) | N fixers at once in `/tmp` worktrees on the batched vLLM server; judge applies the winner | `--` | PLANNED | PENDING | [ ] |
| **EXP-012** | Planned | Fix-Verify Loop | `gemma-4-31b-it-qat` (+E4B) | LoopAgent[fixer, fresh-context critic], K=2–3 | `--` | PLANNED | PENDING | [ ] |
| **EXP-013** | Planned | Repro Test First | `gemma-4-31b-it-qat` (+E4B) | Repro tests in `tests/` (stripped from the patch automatically, so no cleanup) | `--` | PLANNED | PENDING | [ ] |
| **EXP-014** | Planned | Dirty-Tree Insurance | `gemma-4-31b-it-qat` (+E4B) | Keep the best fix applied at all times; harness scores the tree even without submit | `--` | PLANNED | PENDING | [ ] |
| **EXP-015** | Planned | Generation Config Sweep | `gemma-4-31b-it-qat` (+E4B) | Per-role thinking budget / output cap / include_thoughts / temperature | `--` | PLANNED | PENDING | [ ] |

---

## Loophole Programme (EXP-004 – EXP-015, planned 2026-10-30)

Legal levers found in `HARNESS_README.md` and `resources/Competition.md`: things the rules allow but the baseline doesn't use. Each experiment's `notes.md` lists the rule references that make it legal. **Not legal and not planned:** mixing base models (e.g. E4B sub-agents under 31B), which `validate_single_declared_model` rejects.

**Run order (by dependency):**

| Phase | Experiments | Why this order |
| :--- | :--- | :--- |
| 0. Unblock | EXP-004 (Kaggle, user go-ahead), EXP-005 (local spike) | Every later design depends on the time budget and the harness mechanics |
| 1. Cheap, no training | EXP-006, EXP-015, EXP-014, EXP-013, EXP-007 | Prompt/config/skill changes; each can stack on the control |
| 2. Architecture | EXP-009 → EXP-011 vs EXP-012 | Scout first; then compare parallel and sequential use of extra compute at equal wall-clock |
| 3. Training-dependent | EXP-010, EXP-008 | Needs adapters from the Alignment session; EXP-008 is speculative |

**Shared conventions:**
* **CTRL-31B:** single root `LlmAgent` on `gemma-4-31b-it-qat-w4a16-ct` with the current best adapter and sample-submission prompts and sampling. Re-baseline whenever an experiment is adopted.
* **Eval set:** `build/gemma-4-developer-agent/tasks.jsonl` (129 tasks) is for **evaluation only, never training** (and never a source for skill knowledge). **DEV-30** is a fixed 30-task subset stratified by repo, used for fast iteration; the list is still to be created.
* **Noise rule:** with 129 tasks, a difference of ≤ 3 tasks counts as noise unless it repeats with a second seed.
* **E4B:** every experiment marked (+E4B) is re-run separately for the small-model submission. Time profiles differ, so don't transfer eval_config or N between models.

---

## Experiment Index

* [EXP-000: Baseline Zero-Shot Evaluation](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/exp_000_baseline/notes.md)
* *Template:* [experiments/template.md](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/template.md)
* *Schema:* [experiments/template.yaml](file:///d:/FORGE/Research/Gemma%20Fine%20Tuning/experiments/template.yaml)
