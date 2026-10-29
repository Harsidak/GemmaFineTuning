# Experiment Log: [EXP-XXX] - [Experiment Name]

- **Date:** YYYY-MM-DD
- **Author:** [Name]
- **Status:** [PLANNED | RUNNING | COMPLETED | FAILED | ABANDONED]
- **Verdict:** [ADOPT | ITERATE | REJECT]
- **Base Model:** `google/gemma-4-12b-it` / `gemma-4-31b-it-qat-w4a16-ct`
- **Primary Metric (Pass@1):** Baseline: `XX.X%` | Expected: `XX.X%` | Actual: `XX.X%`

---

## 1. Hypothesis & Motivation
* **What are we testing?**
  * *Example:* Fine-tuning Gemma 4 on tool-calling trajectory datasets with explicit chain-of-thought verification tags `<verify>` to reduce premature patch submissions.
* **Why do we expect this to work?**
  * *Example:* The baseline agent failed 42% of tasks due to editing the wrong file before checking existing test cases. Forcing a verification step should improve localization accuracy.
* **Reference Artifacts / Prior Experiments:**
  * Related experiment: `EXP-001`

---

## 2. Configuration & Changes Made
* **Model & Quantization:**
  * Base: `[Model Name]`
  * Precision: `[BF16 / 4-bit QLoRA / QAT]`
* **LoRA Parameters:**
  * $r = 32$, $\alpha = 64$, dropout $= 0.05$
  * Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* **Agent Architecture Changes:**
  * Root `agent.yaml` updates:
  * Sub-agent prompt modifications:
  * Tool additions / constraints:
* **Training Dataset:**
  * Dataset source & size: `N samples`
  * Filtering criteria applied:

---

## 3. Results & Quantitative Evaluation

| Metric | Baseline | Target | Actual | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Pass@1 Rate** | `0.00%` | `0.00%` | **`0.00%`** | `+0.00%` |
| **Localization Accuracy** | `0.0%` | `0.0%` | `0.0%` | `+0.0%` |
| **Patch Validity (Compiles)** | `0.0%` | `0.0%` | `0.0%` | `+0.0%` |
| **Avg Turns per Task** | `0.0` | `0.0` | `0.0` | `-0.0` |
| **Wall-clock Runtime / Task** | `0.0s` | `0.0s` | `0.0s` | `-0.0s` |
| **Timeout / Runaway Rate** | `0.0%` | `0.0%` | `0.0%` | `-0.0%` |

---

## 4. Mistakes Made & Failure Analysis

> [!CAUTION]
> Document all mistakes, unintended behaviors, edge cases, and unexpected errors observed during this run.

### A. Failure Mode Breakdown
1. **Tool-Call Hallucination / Parsing Errors:**
   * *What happened:* (e.g. Agent generated malformed JSON/YAML or called unregistered tools).
   * *Frequency:* `X / N tasks`
2. **Infinite Loops / Budget Depletion:**
   * *What happened:* (e.g. Agent kept repeating `read_file` on the same 10 lines without making progress).
   * *Frequency:* `X / N tasks`
3. **Faulty Bug Localization:**
   * *What happened:* (e.g. Agent edited utility files instead of the core module mentioned in the issue trace).
   * *Frequency:* `X / N tasks`
4. **Patch Application Failures:**
   * *What happened:* (e.g. `git apply` rejected the patch due to indentation mismatch or wrong base commit).
   * *Frequency:* `X / N tasks`
5. **Validation Test Regression:**
   * *What happened:* (e.g. The fix solved the reported issue but broke 3 existing tests).
   * *Frequency:* `X / N tasks`

### B. Specific Task Post-Mortem (Worst Failure Examples)
* **Task ID: `[task_id]`**
  * *Repository:* `[repo_name]`
  * *What went wrong:*
  * *Agent log excerpt:*
    ```text
    [Paste excerpt of agent failure / error traceback]
    ```

---

## 5. Root Cause Analysis (RCA)
* **Why did the mistakes occur?**
  * (Identify the structural or training flaw: e.g. insufficient negative examples during SFT, low learning rate, weak system prompt instructions, lack of tool output truncation).
* **Did the model regress on any previously solved tasks?**
  * Details:

---

## 6. Final Verdict & Decision

- [ ] **ADOPT:** Outperforms baseline significantly, passes validation checks, no major regressions. Ready to package for leaderboard submission.
- [ ] **ITERATE:** Shows promising signal, but identified mistakes must be corrected before full adoption.
- [ ] **REJECT:** Regressed performance or introduced critical bugs (e.g. frequent timeouts, OOM, parse failures). Abandon branch.

### Key Learnings:
1. ...
2. ...

---

## 7. Actionable Next Steps
- [ ] **Immediate Fix 1:** [Describe change to prompt, data, or config]
- [ ] **Immediate Fix 2:** [Describe change]
- [ ] **Next Experiment ID:** `EXP-XXX`
