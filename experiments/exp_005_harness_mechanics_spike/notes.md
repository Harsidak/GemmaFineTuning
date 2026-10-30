# Experiment Log: EXP-005 - Harness Mechanics Spike

- **Date:** 2026-10-30
- **Author:** Research Team
- **Status:** PLANNED (throwaway spike, timeboxed to 1 day)
- **Verdict:** PENDING
- **Base Model:** `gemma-4-e4b-it-qat-w4a16-ct` (the answers don't depend on the model)
- **Primary Metric:** answers to Q1–Q8

---

## 1. Hypothesis & Motivation
* **What are we testing?** The harness behaviours that the docs leave unclear but every multi-agent experiment depends on.
* **Why first:** if we design around a wrong assumption (for example, "sub-agent tool calls are free"), we waste a training or eval run. These are cheap local checks.

| # | Question | Why it matters |
| :--- | :--- | :--- |
| Q1 | Do sub-agent tool calls count against the root `max_tool_calls`? | Budget sizing for EXP-009/011/012 |
| Q2 | Do skill tool calls (`run_skill_script`, `load_skill_resource`) count? | If not, a skill toolchain gives extra tool calls the budget doesn't count (EXP-007) |
| Q3 | Do `ParallelAgent` branches reach vLLM concurrently? | The whole premise of EXP-011 |
| Q4 | Do `git worktree` checkouts in `/tmp` work, and which code does pytest import there? | EXP-011 isolation; the editable install points to `/workspace` |
| Q5 | Does `submit_patch` from a sub-agent end the session? | Only the judge or root may hold `submit_patch` |
| Q6 | Does templating work with `include_contents: none`? Does `{key?}` work? | EXP-009 scout and EXP-012 loop prompts |
| Q7 | Can a `LoopAgent` exit early? | Otherwise a loop always runs `max_iterations` (EXP-012) |
| Q8 | Is a SequentialAgent root rerun in full after a nudge? | Pipeline roots must always end with `submit_patch` |

## 2. Configuration & Changes Made
* Four small throwaway submissions (V1–V4 in `config.yaml`) on 3 tasks with E4B.
* Read the answers from `traces/trace_<id>.json`, the tool-call counters in `task_results.jsonl`, and the vLLM server log.
* Spike code is **not kept**. Only the answers go into this file and `config.yaml`.

## 3. Results

| Q | Answer | Evidence (trace / log line) |
| :--- | :--- | :--- |
| Q1 | `--` | |
| Q2 | `--` | |
| Q3 | `--` | |
| Q4 | `--` | |
| Q5 | `--` | |
| Q6 | `--` | |
| Q7 | `--` | |
| Q8 | `--` | |

## 4. Mistakes Made & Failure Analysis
* (fill in)

## 5. Root Cause Analysis (RCA)
* n/a

## 6. Final Verdict & Decision
- [ ] All questions answered; update the dependent experiments' configs.

## 7. Actionable Next Steps
- [ ] Copy each answer into the "prerequisites" of EXP-007, 009, 011, 012 and 013.
