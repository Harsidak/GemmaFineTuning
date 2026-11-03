# GemmaFineTuning

Fine-tune Gemma 4 into a SWE-bench-style coding agent and package it for the
[Kaggle Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview).

You give the pipeline **agent trajectories** (JSONL conversations in which an agent fixes a bug with the competition's 9
tools). In one time-budgeted session it trains LoRA adapters on them, exports them for the served model, builds a
validated **`submission.zip`** (declarative ADK agent: YAML + prompts + `.safetensors`), scores it on a few competition
tasks, and writes a **`REPORT.md`** that says what happened and what to do next.

```
setup -> preflight -> train:<adapter>... -> export -> package -> eval -> report
 deps     data +       LoRA / QLoRA          bf16      zip +     vLLM +    REPORT.md
 GPUs     leakage      (OOM fallbacks,       safe-     3 vali-   parity +  diagnosis,
 models   gate         resume, best ckpt)    tensors   dators    5 tasks   curves
```

Every phase runs in its own subprocess under a hard deadline counted from the session start, so a hang or crash in
one phase never costs the zip: `package` and `report` always run.

---

## 1. Quick start

### Local machine (one CUDA GPU, >= 16 GiB)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                    # dependencies (+ dev: pytest, ruff)
uv run python -m pytest                                    # ~3 min, CPU only: the whole pipeline on a tiny Gemma-4 fixture

# 1. Check your data (no GPU needed). Fix every ERROR before training.
uv run python pipeline/main.py --action validate-data --preset local_single_gpu --train-data data/my_trajectories.jsonl

# 2. Validate the full config without starting anything.
uv run python pipeline/main.py --action train --preset local_single_gpu --dry-run --train-data data/my_trajectories.jsonl

# 3. Train -> export -> package -> eval -> report.
uv run python pipeline/main.py --action train --preset local_single_gpu \
  --train-data data/my_trajectories.jsonl \
  --base-model /models/gemma-4-e4b-it-qat-q4_0-unquantized \
  --set model.served_path=/models/gemma-4-e4b-it-qat-w4a16-ct
```

Results land in `runs/<UTC date>_<name>/` (`REPORT.md`, `submission.zip`, logs, metrics). Paths that default to
`/kaggle/...` are rewritten to repo-local ones automatically when not on Kaggle (`runs/`, `build/submission.zip`,
`build/gemma-4-developer-agent/tasks.jsonl`); set `MLSFT_PLATFORM=kaggle|local` to force either.

The eval phase needs vLLM and the competition harness (`swegemma`, `adk_submission`; offline wheels in
`build/gemma-4-developer-agent/wheels/`). Without them, add `--set eval.enabled=false`.

### Kaggle (the real runs)

The notebook `kaggle/multi_lora_sft.ipynb` is the launcher. It detects the hardware, finds the attached models and
data under `/kaggle/input`, applies the preset, validates the config in-process and only then starts the
orchestrator. Edit the `RUN` cell (`preset`, `clock_hours`, `smoke_tasks`, ...) and run all cells, or push it from your
machine:

```bash
uv run scripts/kaggle_push.py -multi_lora_sft --hours 11      # 31B on 4x L4 (internet off)
uv run scripts/kaggle_push.py -tpu --hours 2                  # E4B on TPU v5e-8
```

Kaggle imports this repo from a Kaggle **dataset**, not from git: publish a new version of the repo dataset before every
launch. `pipeline/multi_lora_sft/README.md` covers `kaggle_push.py`, wheelhouses, and carrying checkpoints across
sessions.

### Presets

| `--preset` | Config | Model / method | Hardware |
| :--- | :--- | :--- | :--- |
| `31b_4x_l4` (default) | `configs/31b_full.yaml` | 31B NF4 QLoRA, FSDP: all 4 GPUs train at once | Kaggle 4x L4 |
| `e4b_4x_l4` | `configs/e4b_full.yaml` | E4B bf16 LoRA, DDP | Kaggle 4x L4 |
| `e4b_tpu` | `configs/e4b_tpu.yaml` | E4B bf16 LoRA, Tunix (JAX) | Kaggle TPU v5e-8 |
| `e4b_calibration` | `configs/e4b_calibration.yaml` | 30-min E4B speed / memory check | 4x L4 |
| `local_single_gpu` | `configs/local_single_gpu.yaml` | E4B NF4 QLoRA, one GPU, 4 h | your workstation |

Configs live in `pipeline/multi_lora_sft/configs/`; every key not set there comes from `base.yaml` / `config.py`
defaults. (The notebook calls `31b_4x_l4` `31b_gpu_full` and `e4b_4x_l4` `e4b_gpu_full`.)

---

## 2. Training data format

One JSON object per line (`.jsonl`); any number of files. Each record is **one agent run on one task**, in the
OpenAI chat format with tool calls. Check files with `scripts/validate_dataset.py` (or `--action validate-data`): it
lists every problem, shows what each adapter would train on, and runs the same benchmark-leakage gate as preflight.

```json
{
  "task_id": "fastapi__fastapi-1365",
  "agent": "root",
  "repo": "fastapi/fastapi",
  "resolved": true,
  "final_patch": "diff --git a/fastapi/routing.py b/fastapi/routing.py\n...",
  "messages": [
    {"role": "system", "content": "You are a software engineering agent ..."},
    {"role": "user", "content": "<issue text>"},
    {"role": "assistant", "reasoning": "The traceback points at routing.py ...", "content": "",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": {"path": "fastapi/routing.py"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "<file contents>"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "submit_patch", "arguments": {}}}]},
    {"role": "tool", "tool_call_id": "call_2", "name": "submit_patch", "content": "{\"ok\": true}"}
  ]
}
```

| Field | Required | Meaning |
| :--- | :--- | :--- |
| `task_id` (or `instance_id`) | **yes** | Task identifier. The train/val split is by task, so all runs of one task stay on one side. |
| `messages` | **yes** | Non-empty list. Roles: `system`, `user`, `assistant` (or `model`), `tool` (`developer` also accepted). |
| `agent` | **yes** | Which agent of the submission produced this run: `root` (main agent) or a sub-agent name such as `code_analyzer`. Each adapter trains only on records whose `agent` is in its `roles` (`main_lora` → `[root]`). A record without it is never trained on. |
| `resolved` | no (default `true`) | `false` records are dropped while `data.successful_only=true`. |
| `repo`, `base_commit` | no | Used by the leakage gate: a training run on the same repo **and** commit as a held-out task fails preflight. |
| `final_patch` / `patch` | no | The submitted diff; a diff identical to a held-out task's gold patch fails preflight. |
| `tools` | no | Tool schemas shown to the model; default = the tools the template gives that agent role. |
| anything else | no | Kept in your files, ignored by training (`split`, `teacher`, `verification`, ...). |

**Messages**

* `assistant` turns are what the model learns (only their tokens carry loss; the rest is context). A record needs at
  least one. `reasoning` (or `reasoning_content`) is the thinking text; it is rendered before the tool call.
* `tool_calls[i].function.name` should be one of the 9 harness tools: `run_command`, `submit_patch`, `get_status`,
  `read_file`, `edit_file`, `write_file`, `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`.
  `arguments` is a JSON object (or a JSON string of one).
* `tool` messages carry the tool's output, with `name` and the `tool_call_id` of the call they answer.

**What preflight does to your data** (counts are in `REPORT.md` §2)

* Renders each record with the served model's chat template and tokenizes it. Records longer than
  `train.max_seq_length` (16,384 tokens on 31B) are **dropped, not truncated**, as are records that do not render, have
  no trainable tokens, or are exact duplicates.
* Splits off ~5 % of tasks (`data.val_fraction`, max `data.val_max_examples`) as validation.
* **Fails the run** if any record is a held-out competition task (`task_id`, `repo`+`base_commit`, or gold patch match
  against `tasks.jsonl`). Same repo at a different commit is only a warning.
* An adapter with 0 matching records is skipped and its submission slot ships the bare base model; if **no** adapter
  has data, preflight fails and nothing is trained.

**Common errors**

| Validator / preflight message | Fix |
| :--- | :--- |
| `missing 'agent'` / `record(s) are missing the required 'agent' field and 0 matched` | Add `"agent": "root"` to each record produced by the main agent. |
| `adapter 'tool_lora' ... matches 0 records` | No `code_analyzer` data: fine (slot ships bare), or remove the adapter from the config. |
| `BENCHMARK LEAKAGE` | Remove those records. Never train on the competition's `tasks.jsonl` tasks. |
| `no assistant/model turn` | The record has nothing to learn from. |
| `tool_call arguments are not a JSON object` | Write `arguments` as an object (or a valid JSON string). |

---

## 3. What a run produces

```
runs/<run_id>/
  REPORT.md               start here (section 4 below)
  submission.zip          the deliverable (also copied to submission.working_zip_path)
  submission/             the unzipped submission: agent.yaml, sub_agents/, configs/, adapters/<slot>/
  manifest.json           everything machine-readable: config hash, git sha, phase outputs, eval score
  resolved_config.yaml    the exact config this run used
  logs/<phase>.log        full output of each phase subprocess (orchestrator.log = the notebook stream)
  status/<phase>.json     state, timings, error and notes per phase
  metrics/<adapter>.jsonl       one row per logged optimizer step: loss, grad_norm, lr, tokens/s, memory
  metrics/<adapter>_val.jsonl   validation loss per evaluation
  metrics/system.jsonl          CPU / RAM / disk / GPU (or TPU) samples every 30 s
  plots/<adapter>_curves.png    loss + val loss, grad norm, lr, tokens/s
  preflight/report.json   data counts, token stats, leakage report
  checkpoints/<adapter>/  step_N (resumable), best_adapter/, final_adapter/  (removed before eval once bundled)
  adapters/<adapter>/     exported, served-model-ready LoRA
  eval/score.json         smoke-eval tasks, Pass@1, per-task rows; eval/results/ = harness transcripts
```

Exit code: `0` = zip built and setup + preflight passed (eval failures do not fail the run), `1` = setup or preflight
failed (a bare-model zip is still built), `2` = packaging failed.

---

## 4. Reading the results

**While it runs**, the notebook / terminal prints a heartbeat every 3 minutes: phase and session time left, step /
epoch / ETA, loss with its trend, grad norm, lr, tokens/s, GPU utilisation and memory, free disk. During eval it shows
`task 2/5 <task_id> running 4:10 | done PASS 1 FAIL 0 TIMEOUT 0`. A failure before training ends the run with a boxed
`RUN FAILED before training` message naming the phase and error.

**`REPORT.md`**

| Section | Answers |
| :--- | :--- |
| Diagnosis & Next Steps | Everything that went wrong or is worth acting on, most severe first, each with a concrete next step (over-fitting, unused time budget, failed phase, patches not applying, disk, W&B offline, ...). |
| 1. Phase Timeline | Each phase's state, actual vs planned seconds, notes (e.g. base-model load time). |
| 2. Preflight & Data Contract | Records loaded / dropped and why, train/val split, token lengths, leakage result. |
| 3. Adapter Training, Export & Packaging | Steps, epochs, train / val loss, dead-LoRA check, parity, which source each zip slot used. |
| 4. Learning Curves | Validation loss per evaluation next to the train loss, best point marked, plus the PNG. |
| 5. Smoke Evaluation | Pass@1, patch-applied rate, timeouts, tool-call errors, turns, per-task table. |
| 6. Training Health & Resources | Loss first → last, grad-norm range, NaN skips, tokens/s, peak memory, CPU/RAM/GPU/disk peaks. |

**Smoke eval.** After packaging, the eval phase serves the zip with vLLM exactly as the scorer does, checks that the
LoRA changes greedy outputs the way the HF model did (parity), then runs `eval.num_tasks` (default **5**) tasks from
the competition `tasks.jsonl` through the official `swegemma` harness. It catches broken adapters, tool-call format
problems and patches that do not apply. Five tasks cannot rank two models (95 % interval on 2/5 is 5-85 %): compare
models on validation loss and larger evaluation runs. Choose tasks with `eval.task_ids`; the rest are picked
deterministically from distinct repos. Eval is skipped (with the reason in the report) when no adapter was trained
(`eval.allow_bare=true` to evaluate the base model anyway) or less than `eval.min_free_disk_gib` is free.

**W&B** logs online when a `WANDB_API_KEY` Kaggle Secret exists and the machine has internet, else offline into the
run directory (`kaggle_push.py ... -output` uploads it). The key is only read from the environment.

---

## 5. Configuration

Precedence: `config.py` defaults < `base.yaml`-style preset YAML < `--set key=value` (YAML-parsed, repeatable).

```bash
--set train.lr=2e-4 --set 'data.train_files=["a.jsonl","b.jsonl"]' --set 'adapters=[{name: main_lora, roles: [root]}]'
```

| Key | Default (31B) | What it does |
| :--- | :--- | :--- |
| `budget.session_hours` | 12 | Wall clock for the whole session; training gets what export / package / eval / safety margin leave. Notebook: `RUN["clock_hours"]`. |
| `train.max_epochs` / `train.max_steps` | 2 / none | Stop conditions (the time budget also stops training, with a proper LR decay). |
| `train.lr`, `lora.r`, `lora.alpha` | 1e-4, 32, 64 | Optimiser and adapter size. |
| `train.max_seq_length` | 16384 | Longer records are dropped. OOM fallbacks (`train.oom_fallbacks`) retry shorter / other parallelism. |
| `train.val_every_steps` | 8 | Validate every N optimizer steps (plus every `train.val_every_min`). The best-val checkpoint is shipped (`train.select_best`). |
| `train.log_every_steps` | 1 | Metrics row frequency. |
| `adapters` | `main_lora` → `[root]` | One LoRA per entry, trained on records whose `agent` is in `roles`; `overrides` change lora/train per adapter. |
| `eval.smoke_tasks`, `eval.num_tasks`, `eval.task_ids` | true, 5, [] | Competition tasks scored after packaging. |
| `eval.enabled` | true | false = no vLLM step at all (TPU / no harness). |
| `submission.generation.*`, `submission.eval_budget.*` | | Decoding settings and per-task budgets written into the zip. |
| `run.resume_from`, `resume.*`, `output.bundle_dir` | | Continue from a checkpoint / a previous session's bundle. |
| `tuning.enabled` | false | Optuna search over short real trials instead of a training run. |
| `model.prequantized_path` | null | Directory holding the base already in NF4: loads ~3x fewer bytes and multi-threaded (on-the-fly quantization is single-threaded). |
| `model.save_prequantized_to` | null | Save the NF4 base there after the first slow load; publish it as a **private** dataset and set `prequantized_path`. |
| `model.use_liger_kernel` | false | Liger fused RMSNorm / RoPE / MLP kernels (less activation memory). Falls back with a warning if unsupported. |
| `model.attn_implementation` | sdpa | `flash_attention_2` installs the optional wheel, but Gemma 4's 512-wide global heads exceed flash-attn 2's limit (256), so it falls back to sdpa (logged). |

**Fast model load (target < 2 min).** The report's timeline shows `Base model loaded in N s`. To cut it: run once with
`--set model.save_prequantized_to=/kaggle/working/nf4_base`, publish that folder as a private dataset, then run with
`--set model.prequantized_path=/kaggle/input/<that-dataset>`.

**Wheels and monitoring.** `kaggle/build_train_wheels.py` (a Kaggle notebook with internet) builds `train_wheels/`:
peft, trl, accelerate, datasets, wandb, optuna, psutil + nvidia-ml-py (GPU/CPU samples in `metrics/system.jsonl`),
nvitop (`python -m nvitop -1` for a live GPU view), liger-kernel, and flash-attn in `optional/`. It ends with a table of
what was fetched; publish `/kaggle/working/train_wheels` as the new wheels dataset version.

Every key, its validation and its default is in `pipeline/multi_lora_sft/config.py`; the design rationale is in
`pipeline/multi_lora_sft/design_doc.md`.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `RUN FAILED before training: preflight did not pass` | Data contract: run `scripts/validate_dataset.py` on the same files; see `logs/preflight.log`. |
| `train:<adapter>` failed with OOM after all fallbacks | Lower `train.max_seq_length`, or `train.parallelism=model_split` on 31B. |
| Loss NaN / run aborted to last checkpoint | Lower `train.lr`; check for corrupt records. |
| `Eval skipped: No trained adapter was exported` | Something earlier failed; the report's diagnosis names it. |
| `SMOKE TASKS NOT RUN (... snapshots_dir not found)` | Attach the competition dataset (it provides `tasks.jsonl` and `snapshots/`). |
| vLLM fails at startup on 31B | Keep `eval.vllm.max_num_seqs` at 32 and `gpu_memory_utilization` <= 0.90. |
| Free disk low | `train.keep_checkpoints=1`; checkpoints already in the bundle are deleted before eval. |
| Kaggle run uses old code | Publish a new version of the repo dataset; the run's git sha is in `REPORT.md`. |

---

## 7. Competition rules that shape the pipeline

From `build/gemma-4-developer-agent/HARNESS_README.md` (authoritative):

* The submission is **declarative**: root `agent.yaml`, prompts, sub-agents via `agent_tool`, optional LoRA
  `.safetensors`. No Python entrypoints; allowed extensions `.yaml .yml .md .txt .json .safetensors` (`.py` only in skills).
* Allowed models: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`, **one** model for the root and every sub-agent.
* Tools: the 9 listed in section 2. Unpacked submission < 3 GiB; `max_output_tokens`, `thinking_budget` <= 32,768.
* Score: Pass@1, the patch is applied to a clean clone and hidden tests must pass.

---

## 8. Development

```bash
uv run python -m pytest                      # pipeline/multi_lora_sft/tests + tpu_training/tests (configured in pyproject.toml)
uv run python -m pytest -k orchestrator      # a subset
uv run ruff check pipeline tpu_training
```

The tests run every phase for real on CPU with a tiny Gemma-4 checkpoint (FSDP on gloo, export, packaging with the
real validators, eval against a fake harness), so most changes can be verified without a GPU.

| Path | Contents |
| :--- | :--- |
| `pipeline/main.py` | CLI: `status`, `train` (orchestrator by default; legacy `--mode sft/dpo`), `validate-data`, `evaluate`, `package`. |
| `pipeline/multi_lora_sft/` | The orchestrated pipeline: `run.py` (orchestrator), `phases/`, `training/`, `shared/`, `configs/`, `tests/`. |
| `tpu_training/` | Tunix (JAX) trainer for TPU, same checkpoints / reports as the PyTorch path. |
| `kaggle/` | `multi_lora_sft.ipynb` launcher, `build_train_wheels.py` (offline wheelhouses). |
| `scripts/` | `kaggle_push.py` (push / run / download), `validate_dataset.py`. |
| `build/gemma-4-developer-agent/` | Competition harness docs, sample submission, tasks, wheels (not in git). |
| `experiments/` | One folder per experiment with hypothesis and kill criteria; `tracker.md` is the index. |

Never commit weights, `submission.zip`, `runs/` or `logs/`.

## Citation

```bibtex
@misc{harsidak_singh_2026,
  title     = {GemmaFineTuning},
  url       = {https://www.kaggle.com/dsv/21381276},
  DOI       = {10.34740/KAGGLE/DSV/21381276},
  publisher = {Kaggle},
  author    = {Harsidak Singh},
  year      = {2026}
}
```
