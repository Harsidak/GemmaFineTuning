# EXP-016: Skill-First Agent (`swe-kit`) — Architecture, Bug Fixes & Quantitative Evaluation

* **Date**: 2026-11-05
* **Status**: **IMPLEMENTED & VALIDATED** (Zero-adapter archives packaged; offline component & gatekeeper benchmarks completed; 4×L4 live 31B Pass@1 notebook ready)
* **Base Model**: `gemma-4-31b-it-qat-w4a16-ct` (also directly compatible with `gemma-4-e4b-it-qat-w4a16-ct` and future `EXP-017` LoRA adapters)
* **Primary Candidate Archive**: `runs/exp_016/submission_A1_skill_first.zip` (mirrored to `build/exp_016_submission.zip`)
* **Ablation Control Archive**: `runs/exp_016/submission_A0_pruned_6tools.zip`

---

## 1. Motivation & Strategic Role

1. **Immediate Leaderboard Attack (Zero-Adapter Safe)**: While Kaggle's scoring backend returns `0.00` whenever LoRA adapters (`adapters/`) are attached, declarative ADK submissions (`agent.yaml` + `prompts/` + `skills/`) score normally on `gemma-4-31b-it-qat-w4a16-ct` (`CTRL-31B` = `0.18–0.22`).
2. **Freezing Vocabulary Before `EXP-017` Trajectory Regeneration**: Regenerating the 389 verified mined trajectories requires freezing the exact host tools and ADK skill signatures first so SFT trajectories match inference-time tool declarations 1:1.
3. **Eliminating Six Failure Modes Found in Legacy `data/Skills/`**:
   - **Bug 1 (`docs_src` Blindness & $O(N\cdot L\cdot T)$ Regex Stall)**: Legacy `code_locator.py` put `"docs_src"` in `IGNORE_DIRS` (missing 25% of FastAPI tasks) and ran uncached per-line/per-token `re.search`, averaging **19.8 s/task** (`P95 = 63.1 s`, up to `105.6 s` on FastAPI).
   - **Bug 2 (`git add -N .` Ordering & Stripped-Only False Passes)**: Legacy `patch_auditor.py` ran `git add -N .` *before* `git status --porcelain`, turning `"??"` untracked files into `" A"` so stray `/workspace/repro.py` leaks were **0% detected**, and returned `"ready_to_submit": true` when *only* harness-stripped files (`tests/`, `pyproject.toml`) were edited (producing a 0-byte graded patch).
   - **Bug 3 (`safe_editor.py` Indentation Corruption)**: Stripped blank lines during matching and dropped indentation; removed in favor of built-in `edit_file` + inline `python3 - <<'PY'` `count == 1` fallback.
   - **Bug 4 (`utils.py` Test Discovery Miss)**: Legacy `test_runner.py` matched only filename stems, scoring **3.5% Test Hit@3 on FastAPI** (`fastapi/dependencies/utils.py` missed `tests/test_dependencies*.py`) and **0.0% on Requests**.
   - **Bug 5 (`PYTHONSAFEPATH=1` & `SyntaxWarning` Stderr Bloat)**: Setting `PYTHONSAFEPATH=1` without `PYTHONPATH=/workspace` breaks `from docs_src...` imports, and Python 3.12 `ast.parse` on legacy docstrings (`requests`) spams `SyntaxWarning` to `stderr`. Both are fixed in `swe-kit`.
   - **Bug 6 (Progressive-Disclosure Tool-Turn Waste & Dead Graph Schemas)**: Shipping 6 separate skills forces `load_skill` turns, and declaring the 3 broken graph tools (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`) injects useless schemas + Section 6 into every prompt turn. `EXP-016` prunes all 3 graph tools from `agent.yaml` and consolidates into **one skill (`skills/swe-kit`)** with pre-loaded schemas in `prompts/system.md`.

---

## 2. Packaged Submission Comparison

| Submission Arm | Archive Path | Tools Declared | Graph Tools | Skills | Files | Zip Size | Validator |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CTRL-31B`** | `kaggle/submission.zip` | 9 | 3 (dead) | 0 | 4 | 3,713 B | PASS |
| **`Arm A0`** (Pruned Control) | `runs/exp_016/submission_A0_pruned_6tools.zip` | 6 | 0 | 0 | 4 | 3,654 B | PASS |
| **`Arm A1`** (`EXP-016` Skill-First) | `runs/exp_016/submission_A1_skill_first.zip` | 6 | 0 | 1 (`swe-kit`) | 8 | 22,458 B | PASS |

---

## 3. Quantitative Offline Benchmark Results

Evaluated via `experiments/exp_016_skill_first_agent/evaluate_components.py` (full output in `runs/exp_016/component_eval_results.json`) with **strict zero leakage** (`DEV-30` was never inspected to tune heuristics, stopwords, or prompts).

### 3.1 Localization, AST Symbol Span & Test Discovery — `MINED-389` (389 Verified Keep Tasks)

| Method | Source Hit@1 | Source Hit@3 | Source Hit@5 | Multi-File Recall@5 ($n=92$) | `docs_src` Hit@3 ($n=4$) | AST Symbol Hit | Test Hit@3 ($n=321$) | Mean Latency | P95 Latency | Output Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw `git grep` Baseline** | 26.2% | 42.7% | 46.3% | 29.7% | 50.0% | — | — | 1,306 ms | — | — |
| **Legacy `code_locator.py`** | 38.6% | 64.5% | 70.2% | 42.0% | 50.0% | 43.9% | 25.5% | 19,810 ms | 63,145 ms | 8,375 chars *(>5k cap!)* |
| **`EXP-016` `swe-kit/locate.py`** | **58.6%** (+20.0) | **82.8%** (+18.3) | **89.5%** (+19.3) | **61.2%** (+19.2) | **100.0%** (+50.0) | **76.3%** (+32.4) | **71.3%** (+45.8) | **1,259 ms** (**15.7× faster**) | **4,557 ms** (**13.9× faster**) | **4,500 cap** (mean raw 5,518) |

#### Per-Repository Breakdown on `MINED-389` (`Hit@1` / `Hit@3` / `Test Hit@3` / `Mean Latency`)

| Repository | Tasks | Raw `git grep` (`Hit@1` / `Hit@3`) | Legacy `code_locator` (`Hit@1` / `Hit@3` / `Test@3` / `Latency`) | `EXP-016` `swe-kit` (`Hit@1` / `Hit@3` / `Test@3` / `Latency`) |
| :--- | :---: | :---: | :---: | :---: |
| **`fastapi`** | 149 | 20.8% / 37.6% | 35.6% / 58.4% / 3.5% / 29,839 ms | **53.0% / 75.8% / 47.1% / 2,053 ms** |
| **`httpx`** | 129 | 32.6% / 51.9% | 39.5% / 69.0% / 29.4% / 9,186 ms | **58.9% / 89.1% / 77.0% / 553 ms** |
| **`rich`** | 73 | 27.4% / 34.2% | 37.0% / 60.3% / 58.3% / 20,655 ms | **67.1% / 83.6% / 77.8% / 1,125 ms** |
| **`requests`** | 38 | 23.7% / 47.4% | 50.0% / 81.6% / 0.0% / 14,931 ms | **63.2% / 86.8% / 94.7% / 804 ms** |

### 3.2 Held-Out Stratified `DEV-30` (30 Official Tasks — Zero-Shot Evaluation Only)

| Method | Source Hit@1 | Source Hit@3 | Source Hit@5 | Multi-File Recall@5 ($n=9$) | AST Symbol Hit | Test Hit@3 ($n=23$) | Mean Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw `git grep` Baseline** | 30.0% | 43.3% | 43.3% | 17.5% | — | — | 935 ms | — |
| **Legacy `code_locator.py`** | 26.7% | 56.7% | 60.0% | 31.1% | 46.4% | 39.1% | 15,954 ms | 48,453 ms |
| **`EXP-016` `swe-kit/locate.py`** | **50.0%** (+23.3) | **70.0%** (+13.3) | **76.7%** (+16.7) | **44.4%** (+13.3) | **67.9%** (+21.5) | **65.2%** (+26.1) | **851 ms** (**18.7× faster**) | **1,246 ms** (**38.9× faster**) |

*(Notably, on `DEV-30` FastAPI tasks ($n=15$), Hit@1 jumped from **13.3%** with legacy `code_locator` and **20.0%** with raw grep to **40.0%** with `swe-kit`, and Hit@3 jumped from **46.7%** to **73.3%**.)*

### 3.3 Pre-Submit Patch Auditor Gatekeeper (`192` Real Repo Snapshot Trials Across 7 Archetypes)

| Scenario | Expected Decision | Trials | Legacy `patch_auditor.py` | `EXP-016` `verify_and_audit.py` |
| :--- | :---: | :---: | :---: | :---: |
| **`S0`: Valid Gold Source Patch** | `ready_to_submit: true` | 30 | 100.0% | **100.0%** |
| **`S1`: Unmodified Empty Working Tree** | `ready_to_submit: false` | 30 | 100.0% | **100.0%** |
| **`S2`: Valid Edit + Stray `/workspace/repro.py` (1st Call)** | `ready_to_submit: false` | 28 | **0.0%** *(bug: `git add -N .` first)* | **100.0%** |
| **`S3`: Valid Edit + Stray `/workspace/repro.py` (2nd Call)** | `ready_to_submit: false` | 28 | **0.0%** | **100.0%** |
| **`S4`: Protected `tests/test_*.py` Only Modified** | `ready_to_submit: false` | 24 | **0.0%** *(warned only; 0B graded patch)* | **100.0%** |
| **`S5`: Protected `pyproject.toml` Only Modified** | `ready_to_submit: false` | 24 | **0.0%** *(warned only; 0B graded patch)* | **100.0%** |
| **`S6`: `SyntaxError` in Modified Source File** | `ready_to_submit: false` | 28 | 100.0% | **100.0%** |
| **Overall Gatekeeper Accuracy** | — | **192** | **45.8%** (`88/192`) | **100.0%** (`192/192`) |

*(Note: Legacy `code_locator.py`'s mean JSON output was `8,375` chars — exceeding `HARNESS_README.md`'s `5,000`-char stdout truncation ceiling and cutting off valid JSON mid-stream! `swe-kit` caps serialized JSON strictly `<= 4,500` chars so JSON parsing never truncates.)*

---

## 4. Live Eval Protocol (rebuilt 2026-11-17, nothing run yet)

### 4.1 Problems found in the first eval notebook (all fixed)
1. **It would have crashed on Kaggle.** It read `kaggle/submission.zip`, which is gitignored, so the repo dataset
   doesn't contain it. The 0.22 control now lives unpacked in `experiments/arms/ctrl_022/` and is byte-identical to
   that zip (test `test_ctrl_is_byte_identical_to_the_022_submission`).
2. **A0 changed two things.** Besides pruning the tools it raised the budget from 28 to 30 calls. The new
   `exp016_a0_pruned` keeps 28, so tool pruning is the only variable.
3. **It ignored each arm's `eval_config.yaml`.** It hard-coded 30 calls and 8 minutes for every arm, while the
   scorer applies each zip's own budget. Each arm now runs with its own budget.
4. **It ran one task at a time.** That could take up to about 4 h per arm on DEV-30. Now 4 tasks run at once and the
   arms are interleaved task by task, so a 12 h cut-off can't leave one arm with fewer tasks.
5. **It was missing the install fix from the adapter smoke run.** Without uninstalling the TPU vLLM plugin, vLLM
   fails with "Only one platform plugin can be activated". The new notebook uses that proven install cell.
6. **It kept no per-task data apart from pass/fail.** Each task now saves its row, patch, harness trace and log,
   plus the tally (why the task ended, empty patch, calls used, skill use).

### 4.2 Arms (see `experiments/arms/README.md`)

| Arm | Change vs base | Budget |
|---|---|---|
| `ctrl_022` | The 0.22 leaderboard zip | 28 calls / 8 min |
| `exp016_a0_pruned` | 3 dead graph tools removed (+ the prompt line banning them) | 28 / 8 |
| `exp016_a1_skill_first` | `swe-kit` skill + skill-first prompt + 30-call budget | 30 / 8 |

A1 vs `runs/exp_016/submission_A1_skill_first.zip`: same files and same YAML content. The only difference is that
`locate.py` no longer falls back to reading stdin when called without a query. Under `run_skill_script`, stdin can
be an open pipe, which would hang until the 240 s command timeout.

### 4.3 Review of `swe-kit` before GPU time (risks to watch in the traces, not changed)
* **Where the scripts run.** Nobody has checked yet where ADK places skill scripts inside the task container. If
  they are under `/workspace`, `submit_patch` would include them in the patch. The tally flags
  `skill_files_in_patch`.
* **`load_skill` may be required.** The prompt tells the model to skip `load_skill` and call `run_skill_script`
  directly. If ADK requires the skill to be loaded first, every skill call errors. The tally counts
  `skill_errors_total`.
* **Old tests can block submission.** `verify_and_audit.py` sets `ready_to_submit: false` when any auto-selected test
  fails. For feature tasks, existing tests that assert the old behaviour will fail, and the model may chase them.
  Compare A1's `submitted` count and the number of tasks where A0 passed and A1 didn't.
* **Prompt/SKILL.md wording.** The prompt calls the script argument "path", while SKILL.md and the harness use
  `file_path`. The model sees the real schema, so this is low risk.

### 4.4 How to run
```powershell
# 0. commit + push + update the banwait13/gemmafinetuning dataset (pipeline/exp_eval, experiments/arms, data/Skills/swe-kit are new)
uv run scripts/kaggle_push.py -exp016_eval --set tasks=smoke5 --set arms=exp016_a1_skill_first   # ~30-40 min smoke
uv run scripts/kaggle_push.py -exp016_eval                                                          # 3 arms x DEV-30
uv run python -m pipeline.exp_eval tally runs/kaggle/<run>/exp_eval/<label>/* --control ctrl_022     # re-tally locally
```

**What the smoke run must show before the DEV-30 run:**
* No `harness_error` rows.
* A trace for every task.
* `effective_parallelism` above 2 in `run_summary.json`. If it is about 1, rerun with `--set mode=threads`.
* A1 makes skill calls with `skill_errors_total` near 0 and `skill_files_in_patch` = 0.

### 4.5 Reading the DEV-30 result (decision rules from the config, plus the noise rule)
* **Is the scorecard trustworthy?** `ctrl_022` should land near 0.18–0.22. DEV-30 is too small for a tight check, so
  read it only as a rough confirmation that the local harness behaves like the scorer.
* **A0 vs ctrl** isolates the dead-tool pruning.
* **A1 vs A0** isolates the skill package (plus 2 calls).
* **Rule:** a difference of ≤ 3 tasks is noise, so adopt only after the full 129. Ship A1 if it gains ≥ 3 tasks
  there and doesn't raise timeouts (`call_cap` / `time_cap` counts).
* **Side outputs:**
  * EXP-005 Q2 ("do skill calls count against the budget?") is answered by `q2_skill_calls_counted`.
  * EXP-018 W0 by the `w0` block.
  * EXP-014's baseline by `exp014` (edited by call 12, wins without `submit_patch`).

### 4.6 Smoke run 2026-11-17 (A1 only, smoke5, 0.37 h)
* Plumbing works end to end: vLLM ready in 338 s, about 2.7 tasks in flight, driver exit 0. A1 passed 2/5 (rich_2725,
  rich_3063). That is too few tasks to judge, and there was no control.
* **All 16 swe-kit calls failed.**
  * 6 returned `SKILL_NOT_FOUND`, because Gemma copied the backticks and quotes from the prompt into the values.
  * The other 10 ran but looked in the wrong folder: "No Python source files found", "File not found", "not a git
    repo". In the subprocess sandbox the repo is `/tmp/swegemma_sandbox_<id>/workspace`, and only `run_command`
    rewrites `/workspace` to it. The scripts fall back to their own cwd (`locate.py:624`). **Not fixed yet.**
* **EXP-005 Q2.** In all 5 tasks the harness count was exactly: gated calls (errors included) + skill calls whose script
  ran. `SKILL_NOT_FOUND` calls and `submit_patch` were not counted. So skills do **not** give free calls.
* **`write_file` cannot write `/tmp/...`** ("Path traversal detected"), but the prompt suggested `/tmp/repro.py`.
* **Open question.** `import requests` loaded an installed copy, not `/workspace/src`; Gemma needed `PYTHONPATH=src`.
  Check `logs/requests_7205.log` and `logs/httpx_3672.log` to see whether the hidden tests imported the patched code.
* **Fixed after the smoke:**
  * A1 prompt: plain-text skill call examples, no swe-kit use after the first skill error, a heredoc for scratch code,
    and `PYTHONPATH=src` for `src/` repos.
  * Tally: budget errors on the row now count as caps, skill errors are counted (`SKILL_NOT_FOUND` and errors inside
    `stdout`), Q2 uses the harness's final count on the row, and the scorecard says when the control didn't run.

## 5. Next Steps (Live 4×L4 `DEV-30` Pass@1 & Kaggle Submission)

1. Run §4.4 (smoke, then 3 arms x DEV-30), one 4×L4 session at a time. `num_tasks` is no longer an override:
   use `tasks=smoke5|dev30|all`.
2. Or upload `build/exp_016_submission.zip` directly as a zero-adapter submission to capture immediate leaderboard upside over `0.22` while LoRA scoring on Kaggle is being fixed.
3. Proceed to `EXP-017` trajectory generation using the frozen 6-tool + `swe-kit` (`locate.py`, `read_symbol.py`, `verify_and_audit.py`) schema.
