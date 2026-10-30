# Experiment Log: EXP-015 - Generation Config Sweep

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED
- **Verdict:** PENDING
- **Base Model:** `gemma-4-31b-it-qat-w4a16-ct` (sweep E4B separately)
- **Primary Metric:** Pass@1 and wall-clock per task (we care about Pass@1 per minute)

---

## 1. Hypothesis & Motivation
* **What are we testing?** Per-agent `generate_content_config`:
  * `thinking_budget`
  * `max_output_tokens`
  * `include_thoughts`
  * temperature
* **Why it matters:**
  1. **Time:** thinking tokens are the most expensive part of each turn, and under a tight cap (EXP-006) every saved second goes to more tool calls or more best-of-N candidates (EXP-011).
  2. **Context:** with `include_thoughts: false` the model still thinks on each turn, but old thoughts are dropped from the history. The context fills more slowly and compaction happens less often.
  3. **Truncation:** the harness has a dedicated nudge for tool calls cut off at `max_output_tokens` (§5.3). Every truncation wastes a turn.
* **Per role, the best settings probably differ:**
  * scout: little thinking, short output;
  * fixer: more thinking;
  * judge: little thinking, temperature 0.

### Kill criteria
* No setting beats the control by at least 1 task or saves at least 20% of wall-clock on DEV-30 → keep the control.

## 2. Configuration & Changes Made
* Vary one factor at a time from the control on DEV-30 (11 runs, fixed seed), combine the best value of each, then confirm the top 2 on all 129.
* Leave `thinking_level` unset (vLLM + LoRA serving).

## 3. Results & Quantitative Evaluation

| thinking_budget | max_out | include_thoughts | temp | Pass@1 (DEV-30) | Wall-clock / task | Truncation nudges / task | Compactions / task |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4096 | 16384 | true | 0.0 | `--` (control) | `--` | `--` | `--` |
| 0 | 16384 | true | 0.0 | `--` | `--` | `--` | `--` |
| 1024 | 16384 | true | 0.0 | `--` | `--` | `--` | `--` |
| 8192 | 16384 | true | 0.0 | `--` | `--` | `--` | `--` |
| 4096 | 8192 | true | 0.0 | `--` | `--` | `--` | `--` |
| 4096 | 16384 | false | 0.0 | `--` | `--` | `--` | `--` |
| 4096 | 16384 | true | 0.3 | `--` | `--` | `--` | `--` |
| 4096 | 16384 | true | 0.6 | `--` | `--` | `--` | `--` |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* (fill in)

## 6. Final Verdict & Decision
- [ ] **ADOPT** chosen per-role configs / [ ] **REJECT** (keep control)

## 7. Actionable Next Steps
- [ ] Feed the chosen configs and their time profile back into EXP-006 pacing.
