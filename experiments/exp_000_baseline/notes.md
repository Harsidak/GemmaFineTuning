# Experiment Log: EXP-000 - Baseline Zero-Shot Evaluation

- **Date:** 2026-10-16
- **Status:** COMPLETED
- **Verdict:** BASELINE ADOPTED (Reference Point)
- **Base Model:** `google/gemma-4-12b-it` (4-bit quantized)
- **Primary Metric (Pass@1):** Actual: `14.0%` (7 / 50 tasks solved)

---

## 1. Objective & Hypothesis
Establish an initial benchmark score using un-tuned Gemma 4 12B Instruction-tuned model paired with default ADK tools (`read_file`, `edit_file`, `run_command`, `submit_patch`).

## 2. Quantitative Results
* Pass@1: **14.0%** (7 / 50 tasks)
* Average turns per task: 14.8 turns
* Timeout / Runaway Rate: 10.0% (5 tasks hit max turn ceiling of 20)

## 3. Top Mistakes & Failure Modes Observed
1. **Premature Editing:** Model frequently edited files after skimming only the first 20 lines without checking function definitions or imports.
2. **Infinite `read_file` loops:** When encountering unfamiliar repository layouts, the model re-read the same directory files 4-6 times without progressing.
3. **Missing Validation Check:** In 19 failed tasks, the model introduced syntax or import errors that could have been caught by running `pytest` via `run_command`, but it called `submit_patch` immediately instead.

## 4. Root Cause Analysis
Zero-shot prompting lacks tool-usage discipline and task-phase awareness. Without fine-tuning on high-quality agent trajectories, Gemma 4 is prone to premature termination and hallucinates file paths.

## 5. Next Steps
1. Prepare trajectory training dataset formatted with Gemma 4 special tokens.
2. Fine-tune LoRA adapter focusing on tool call precision and step-by-step verification.
3. Split agent into specialized roles (`Localizer` -> `Patcher` -> `Verifier`).
