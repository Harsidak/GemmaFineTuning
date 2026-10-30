# Multi-LoRA SFT Pipeline: Design Document

| | |
|---|---|
| Status | Draft for review. No code exists yet. |
| Date | 2026-10-29 |
| Location | `pipeline/multi_lora_sft/` |
| Background reading | `resources/lora_sft_playbook.md` explains the reasoning and sources behind most decisions below. Read §2–§8 of it first. |
| Authoritative rules | `build/gemma-4-developer-agent/HARNESS_README.md` and `build/gemma-4-developer-agent/sample_submission/` |

This document describes what to build. Someone who knows PyTorch, Hugging Face (transformers, PEFT, TRL, Accelerate) and Kaggle should be able to build the pipeline from it without asking the author anything. Decisions are marked so you can tell who made them:

- **[decided]**: the project owner chose this. Do not change it without asking.
- **[design call]**: the author chose this as a sensible default. Change it if you find a better reason, and record why in this doc.
- **[builder]**: deliberately left to you.
- **[verify]**: a fact we have not confirmed. Check it on the first real run and update this doc.

---

## 1. Why this exists

We are competing in the Kaggle **Gemma 4 Developer Agent** competition. A submission is a `submission.zip` that holds:

- a declarative ADK agent (YAML and Markdown prompts);
- optional LoRA adapters (`adapters/<name>/adapter_model.safetensors`).

The scorer serves one Gemma 4 base model with vLLM on 4×L4 GPUs. It runs the agent on hidden SWE-bench-style tasks and counts how many pass (Pass@1).

Data collection, prompts and agent design are owned by other tracks. **This pipeline turns trajectory data into a ready-to-submit zip in one unattended Kaggle session.** Once it exists, improving the model needs only two kinds of work:

1. collecting better data;
2. pressing "run" on Kaggle.

### What success looks like

One Kaggle notebook run, started with a config file:

1. trains LoRA adapters for **one** base model for most of the 12-hour session (about 10 h of actual training);
2. writes `submission.zip` that passes every validation check;
3. in the final hour, runs the real agent on **3 fixed tasks** from the competition's 129 build tasks, with the same vLLM stack the scorer uses, and prints the exact score;
4. leaves a report and manifest that say what was trained, on what data, with what result, and what went wrong if anything did.

If the run crashes, runs out of time or hits a bad config, it must **still** produce a valid `submission.zip`, using the best adapter it has. The report must say which one it used.

---

## 2. Goals, non-goals, anti-goals

### Goals

- **G1. One command per run.** A notebook cell calls `python -m pipeline.multi_lora_sft.run --config <yaml> [--set key=value ...]`. Everything else is automatic.
- **G2. Time-aware.** The whole run is planned against the Kaggle 12-hour wall clock, not against epochs. Training stops in time for export, packaging and eval, with margin.
- **G3. Multi-LoRA.** The config holds a list of adapters. Each has a name, a filter over the data by agent role, and optional hyperparameter overrides. The two standard setups are:
  - `[main_lora, tool_lora]` **[decided]**: `main_lora` for the root `swe_baseline_agent`, `tool_lora` for the `code_analyzer_agent` sub-agent;
  - `[shared]`: one adapter for both roles, used as the comparison baseline.
- **G4. Always ship.** `submission.zip` is built *before* the eval starts, and is rebuilt from a fallback if training failed.
- **G5. Fail fast.** Everything that can be checked on CPU is checked in the first ~15 minutes, before any GPU-hour is spent. That covers config, data, template, leakage and tool-call format.
- **G6. Train/serve parity is measured.** The run reports whether vLLM serving our adapter behaves like the model we trained (playbook §4).
- **G7. Resumable.** Wall-clock checkpoints let a killed session continue in the next session.
- **G8. Reproducible.** Every run records its resolved config, data hashes, template hash, code hash, library versions and seeds.

### Non-goals (v1)

- Writing or tuning prompts, skills, `agent.yaml` or sub-agent YAMLs. The pipeline **copies** a submission template supplied by the agent track. It only changes `model:` names and inserts adapters.
- DPO / ETO (stage 2). It will reuse much of this pipeline later, but it is not in scope here.
- Data collection or trajectory generation.
- Training several base models in one run, or sweeping several configs in parallel on separate GPUs. Keep the design open to a later "sweep mode" (four E4B configs, one per GPU), but do not build it now.
- Windowing long trajectories into post-compaction views (playbook §6). v1 drops over-long trajectories and reports how many it dropped.
- A leaderboard-quality score. Three tasks are a smoke test (see §9.4).

### Anti-goals (the run is a failure even if it "works")

- **Training on any of the 129 build tasks.** These tasks are our only test set. Leakage invalidates every score after it.
- **Hardcoded private data paths in code.** All mined data lives in the gitignored `DataMining/`. Data paths come only from config or CLI flags. Code must never contain `DataMining` paths or private dataset slugs.
- **A run that ends without a valid `submission.zip`.**
- **A score from a different stack than the scorer's.** Using SGLang, a different vLLM version, a different template or a transformers-only generation for the eval invalidates the score.
- **Silent degradation.** Examples: dropping half the data, dead adapters, template mismatch, a run that never decayed its LR. Each must appear in the report.

---

## 3. Constraints

### 3.1 Kaggle session

| Constraint | Value | Source |
|---|---|---|
| Machine | `machine_shape: "NvidiaL4"` = 4× L4, 24 GB each, bf16, SM89 | official starter notebook **[verified]** |
| Internet | **Off** on L4 sessions. Every package must come from attached datasets. | competition overview **[verified]** |
| Session limit | 12 h hard kill. A killed session may lose unsaved output. | Kaggle |
| Quota | L4 counts 2× against ~30 h/week, so about **one full run per week** | competition overview **[source]** |
| Docker image | pinned digest from the starter's `kernel-metadata.json` | starter **[verified]** |
| Output disk | `/kaggle/working` is ~20 GB and kept as the notebook output | Kaggle **[verify]** current limit |

**Assumption [design call]:** we get about one full L4 run per week, plus short calibration runs. If more quota appears, nothing in this design changes; we just run more often.

### 3.2 Software stack

The scorer uses the official wheelhouse dataset `metric/gemma-4-developer-agent-wheelhouse`:

- `vllm==0.19.1`, `transformers==5.13.1`, `compressed_tensors==0.15.0.1`, `bitsandbytes==0.46.1`;
- `swegemma==0.2.11`, `adk_submission==0.2.13`, `adk_eval_core==0.1.0`, `google_adk==1.36.1`.

It ships **no** torch (torch comes from the docker image), peft, trl, accelerate or datasets.

So we need our **own private training wheelhouse dataset**, pinned to be compatible with `transformers==5.13.1` and the image's torch. Training and eval run in **one Python environment** **[design call]**:

1. install the official wheelhouse first;
2. then install our training wheels with `--no-deps`;
3. the setup phase checks that both stacks import.

Building that wheelhouse is part of this work (§12, step 1).

### 3.3 Submission rules (from `HARNESS_README.md`)

- **Models.** Exactly one base model across all agents, chosen from `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`.
- **Adapters.** Each lives at `adapters/<name>/{adapter_config.json, adapter_model.safetensors}`, with names matching `[A-Za-z0-9][A-Za-z0-9._-]*`. Rank must be ≤ 128.
- **Size.** Total unpacked size < 3 GiB (3,221,225,472 bytes).
- **Extensions.** Only `.yaml .yml .md .txt .py (skills only) .json .safetensors` are allowed. No symlinks, no path traversal, no `*.py` at the root.
- **Token limits.** `max_output_tokens` and `thinking_budget` must be ≤ 32,768.
- **Adapter key format.** Keys look like `base_model.model.model.language_model.layers.{i}.<module>.lora_{A,B}.weight`, as in the sample adapters.

### 3.4 Model facts that shape the code (playbook §2)

| | E4B | 31B |
|---|---|---|
| Text layers / hidden size | 42 / 2560 | 60 / 5376 |
| Vocabulary | 262,144 | 262,144 |
| `final_logit_softcapping` | 30.0 | 30.0 |
| Head dim, sliding / global layers | 256 / **512** | 256 / **512** |
| Special | 18 KV-shared layers; per-layer embeddings | — |
| Architecture class | `Gemma4ForConditionalGeneration` (has vision/audio towers) | same (vision only) |

What this means for the code:

- **Huge vocabulary.** Full logits are 8.6 GB in bf16 for 16k tokens. The loss must not build the full logits (§7.3).
- **Head dim 512.** FlashAttention-2 does not cover it. Use `attn_implementation="sdpa"` unless you verify otherwise **[verify]**.
- **Vision tower.** Plain module names like `q_proj` also match the vision tower. LoRA target modules must be a regex pinned to `language_model`.
- **Softcap.** Any custom loss must apply `30 * tanh(logits / 30)` before the cross-entropy.

### 3.5 Project rules

- Code is under `pipeline/` and must import as `pipeline.…` from a Kaggle Dataset copy of the repo. Do not depend on anything outside `pipeline/`.
- Do **not** modify `pipeline/src/training/sft_trainer.py` or `pipeline/configs/train_lora_sft.yaml`; the Alignment track owns them. This pipeline is self-contained, but it **reuses** these:
  - `pipeline/src/data/processor.py`: `tokenize_trajectory`, `load_chat_template`, `normalize_messages`, `TemplateMismatchError`, `IGNORE_INDEX`. This is the assistant-only loss mask, built from the chat template.
  - `pipeline/src/data/loader.py`: `load_tasks`, `load_trajectories`.
  - `pipeline/src/utils/packager.py`: `validate_submission_directory`, `package_submission_archive`.
  - `pipeline/src/agent/tools.py`: `OFFICIAL_HOST_TOOLS`.
- If you need a change in a shared module, add the new behavior inside `multi_lora_sft/` instead, or ask the module's owner.
- Never commit weights, `submission.zip`, `runs/` or `logs/`.

---

## 4. Decisions already made

| # | Decision | Status | Why (playbook section) |
|---|---|---|---|
| D1 | Role-split adapters: `main_lora` (root coder) and `tool_lora` (code analyzer). `shared` is supported as a baseline. | [decided] | §12 |
| D2 | LoRA r=32, α=64 by default. The parity check (§9.3) decides whether r=32 is safe on vLLM's int4 path. If not, fall back to r=16, α=32. | [decided] r/α; [design call] fallback | §5, §4.3 (vllm#50059) |
| D3 | 31B is the main model; E4B is the small-model track and the cheap test bed. | [decided] | §13 |
| D4 | Stage 1 is SFT with assistant-only loss. Stage 2 (DPO) is out of scope. | [decided] | §6 |
| D5 | The backbone is **TRL + PEFT + Accelerate**, not Unsloth. The original plan said Unsloth, but on 4×L4 without internet it is the riskier choice. Keep a `backend` config field so a single-GPU Unsloth backend can be added later. | [design call] — **owner please confirm** | §11 |
| D6 | No inference server during training. The eval uses the official vLLM 0.19.1 through `adk_submission.VllmServer`, exactly as the starter notebook does. | [design call] | §10, §1.3 |
| D7 | The training text is rendered with the **served** model's chat template (from the `-qat-w4a16-ct` model folder), never with the training checkpoint's template. | [design call], high confidence | §4.1 |
| D8 | Training weights are the QAT bf16 parent (`*-it-qat-q4_0-unquantized`). E4B trains in bf16; 31B is loaded as NF4 (QLoRA) because bf16 does not fit. | [design call] | §4.2, §3 |
| D9 | Epochs: up to `max_epochs` (default 2, from the owner's "1.5–2 epochs"), or until the time budget runs out, whichever comes first. | [decided] range; [design call] mechanism | §5 |
| D10 | Extra loss weight on tool-call tokens (owner's plan: 1.5×). It is a config knob, **default 1.0**, and should be A/B-tested. | [decided] knob exists; [design call] default — **owner please confirm** | §6 |
| D11 | Learning rate 1e-4 by default, with a time-triggered WSD schedule. | [design call] | §5 |
| D12 | `submission.zip` is built **before** the eval, so an eval crash or session kill cannot cost us the submission. | [design call] | §8 |

---

## 5. How a run is laid out in time

All times are measured from **session start** `T0`. The notebook's first cell records `T0` and passes it in, because the 12 h limit includes wheel installs.

```
T0 ───────────────────────────────────────────────────────────────────────────── T0+12:00 (Kaggle kill)
│setup│preflight│              train (adapter 1, then adapter 2)        │exp│pkg│   eval   │margin│
│≤10m │  ≤15m   │        ends at train_deadline = T0 + 10:25            │10m│5m │   ≤60m   │ 20m  │
```

| Budget key | Default | Meaning |
|---|---|---|
| `budget.session_hours` | 12.0 | Kaggle's hard limit |
| `budget.safety_margin_min` | 20 | Never planned for; absorbs overruns |
| `budget.eval_min` | 60 | vLLM start-up (31B with TP=4 can take up to 20 min) plus 3 tasks |
| `budget.package_min` | 5 | |
| `budget.export_min` | 10 | |
| `budget.setup_max_min` | 10 | Wheel install and environment check. The phase fails if it takes longer. |
| `budget.preflight_max_min` | 15 | |

How the deadlines are computed:

- `train_deadline = T0 + session − margin − eval − package − export`. With the defaults that is **T0 + 10h25m**.
- Training actually starts after setup and preflight, so it gets about **10 h**.
- The owner asked for "about 11 h". Reaching that means shrinking the margin and eval budgets, which is possible once real timings are measured. The report prints each phase's planned and actual duration, so we can tighten them with data.

**Rule [design call]:** each phase gets a hard deadline. The orchestrator enforces it by killing the phase's whole process group, not by trusting the phase to stop itself. Phases also check the clock themselves and stop cleanly before the hard kill.

---

## 6. Architecture

### 6.1 Phases

Each phase runs as a **separate subprocess** **[design call]**. This has three benefits:

- GPU memory is guaranteed free between training and vLLM.
- A crash in one phase cannot corrupt another phase's outputs.
- Timeouts are enforceable.

```
run.py (orchestrator: plans the clock, launches phases, decides fallbacks, writes the report)
  │
  ├─ 1. setup      install/verify wheels, check GPUs, check paths                 (CPU)
  ├─ 2. preflight  validate config + data, check leakage, tokenize, cache, project  (CPU)
  ├─ 3. train      once per adapter, in config order, with resume                   (GPU)
  │                 ends by saving: final adapter + HF parity reference
  ├─ 4. export     convert/verify adapters into submission format                   (CPU)
  ├─ 5. package    stage the submission template + adapters, validate, zip          (CPU)  ← always runs
  ├─ 6. eval       vLLM server + parity check + 3-task smoke eval                    (GPU)
  └─ 7. report     REPORT.md + manifest.json, print the score                        (CPU)  ← always runs
```

Each phase reads its inputs from the run directory and writes `status/<phase>.json`:

```json
{"phase": "train:main_lora", "state": "ok|failed|timeout|skipped", "started": "...", "ended": "...",
 "planned_deadline": "...", "error": null, "outputs": {"adapter_dir": "..."}, "notes": ["..."]}
```

### 6.2 What happens when something fails

The orchestrator never stops early. It always reaches `package` and `report`.

| Failure | Action |
|---|---|
| `setup` fails (missing wheel, wrong GPU) | Skip train and eval. Package from the fallback adapter (below). Report the error. |
| `preflight` fails a **hard** check (leakage, schema, template mismatch above threshold, no data for an adapter) | Skip training that adapter. Other adapters continue. |
| Out of memory in the first training steps | Retry with the next entry in `train.oom_fallbacks`, e.g. `[{quantization: nf4}, {max_seq_length: 12288}]`. Log each attempt. |
| Train crashes mid-run | Restart from the last checkpoint if at least `train.min_restart_min` (default 20) remains. At most `train.max_restarts` (default 2) restarts. |
| Train reaches its deadline | Normal: it decays the LR, saves and exits cleanly. |
| Export fails for an adapter | Use that adapter's latest checkpoint, converted the same way. If that also fails, use the fallback. |
| Package validation fails | Fail loudly in the report and keep the staging directory for inspection. This should never happen in a tested build. |
| Eval fails or times out | The score is reported as "not available" with the reason. `submission.zip` is unaffected because it was already built. |

**Choosing an adapter for packaging [design call]** — for each submission adapter name, use the first that is available:

1. this run's exported final adapter;
2. this run's newest checkpoint;
3. `submission.fallback_adapters_dir/<name>`, e.g. the previous best run's output attached as a dataset;
4. no adapter. The agent YAML's `adapter:` line is then removed, so the agent runs on the bare base model. This is allowed, and it is better than shipping nothing.

The manifest records which source was used.

### 6.3 Code layout

```
pipeline/multi_lora_sft/
├── design_doc.md            # this file
├── README.md                # how to run it (short; written when the code exists)
├── __init__.py
├── run.py                   # orchestrator CLI (entry point)
├── config.py                # dataclass schema, YAML load, --set overrides, validation
├── clock.py                 # SessionClock: T0, deadlines, remaining time, phase budgets
├── runctx.py                # run dir layout, status files, manifest, logging setup
├── phases/
│   ├── setup.py
│   ├── preflight.py
│   ├── train.py             # one adapter per invocation; launched via accelerate for DDP
│   ├── export.py
│   ├── package.py
│   ├── evaluate.py
│   └── report.py
├── training/
│   ├── data.py              # role filter, split, tokenize (via processor), cache, length-grouped sampler
│   ├── model.py             # load base, attach LoRA by regex, gradient checkpointing, device placement
│   ├── loss.py              # selective-logits CE with softcap and optional per-token weights
│   ├── schedule.py          # time-aware WSD LR schedule
│   ├── callbacks.py         # time budget stop, wall-clock checkpoint, dead-adapter check, metrics
│   └── parity.py            # HF-side parity reference (greedy tokens + logprobs)
├── configs/
│   ├── base.yaml            # every field with its default and a comment
│   ├── e4b_calibration.yaml # 30-min run: measure memory + tokens/s, end to end
│   ├── e4b_full.yaml
│   └── 31b_full.yaml
└── tests/                   # CPU-only pytest suite (see §11)

kaggle/multi_lora_sft.ipynb  # 4-cell launcher (see §10), at the repo root next to the other launchers,
                             # because scripts/kaggle_push.py pushes notebooks from kaggle/
```

Module names are a suggestion **[builder]**. Keep the split between orchestrator, phases and training internals, though: it is what makes phase isolation and testing work.

---

## 7. Training in detail

### 7.1 Data contract (input)

The input is one or more JSONL files given by `data.train_files` (paths only from config or CLI). Each line is one trajectory:

```json
{
  "task_id": "string, unique per task",
  "repo": "owner/name",
  "base_commit": "sha (optional but needed for the leakage check)",
  "resolved": true,
  "agent": "root | code_analyzer",
  "messages": [ {"role": "user|assistant|model|tool|system", "content": "...", "tool_calls": [...], "reasoning": "..."} ],
  "tools": [ optional OpenAI-format tool schemas exactly as the harness sent them ],
  "source": "optional free text: which miner/generator produced it"
}
```

- **Messages.** `messages` uses the same shape `processor.normalize_messages` already accepts: `model` is mapped to `assistant`, and JSON-string arguments are parsed. Example: `data/train_trajectories.jsonl`.
- **`agent` field.** It is **new** and is required whenever an adapter has a role filter.
  - The analyzer sub-agent's conversations are separate sessions in the harness (`agent_tool` with `skip_summarization: true`). So each analyzer session is its own record with `agent: code_analyzer`.
  - The Data track produces this field. Tell them about it.
- **`tools`.** If it is missing, fall back to `processor.default_tools()`. Better: use the tool list for that role from the submission template (the analyzer only has 4 tools). A model trained with 9 tools in its system prompt but served with 4 sees a different context.
- **Unresolved trajectories.** Skipped when `data.successful_only: true` (the default).

### 7.2 Preparing the data (preflight phase)

For each adapter:

1. **Load and filter.**
   - Keep records whose `agent` is in the adapter's `roles`.
   - Drop `resolved: false` (when configured).
   - Deduplicate on a hash of the rendered text.
2. **Split.** Hold out about 5% **by `task_id`**, deterministically via a hash of `task_id` and the seed, never by turn. Cap the validation set at `data.val_max_examples` (default 64) so a validation pass takes a few minutes, not tens.
3. **Tokenize.**
   - Call `processor.tokenize_trajectory(messages, tokenizer, tools, chat_template=<served template>, max_length=train.max_seq_length, **data.template_kwargs)`.
   - It returns `None` for trajectories that are too long; count them as dropped.
   - Count `TemplateMismatchError`s separately.
   - Use a `multiprocessing` pool. The tokenizer comes from the **served** model folder.
4. **Tool-call weights** (only if `loss.tool_call_weight != 1.0`).
   - Within supervised tokens, every token from `<|tool_call>` through `<tool_call|>` (inclusive) gets that weight; all other supervised tokens get 1.0.
   - Find the spans by token id. Both markers are special tokens, so no change to `processor.py` is needed. Confirm the token ids from the tokenizer **[verify]**.
5. **Cache** the result as Arrow or Parquet (`input_ids`, `labels`, optional `weights`, `length`, `task_id`) under `runs/<run_id>/cache/<adapter>/`.
   - Key the cache by a hash of: data file hashes, template hash, tokenizer hash, `max_seq_length`, `template_kwargs`, roles and seed.
   - A matching cache from an attached dataset (`data.cache_dir`) is reused instead, so preflight can also run once on the dev machine.

### 7.3 Loss

The loss is a custom `compute_loss` that never builds full logits **[design call]**:

1. Run the decoder to get final hidden states `h` of shape `[B, T, H]`. These come after the final norm and before `lm_head`.
2. Shift by one position. Select the rows where `labels != -100`, giving `h_sel` of shape `[N, H]`.
3. In chunks of `loss.chunk_rows` (default 4096):
   - compute `logits = lm_head(h_sel_chunk)`;
   - convert to fp32 and apply `logits = 30 * tanh(logits / 30)`, reading the cap from `config.final_logit_softcapping`, not a hardcoded value;
   - compute the per-token cross-entropy.
4. Combine as `loss = Σ(w_i · ce_i) / Σ(w_i)` over **all supervised tokens in the optimizer step**, not per micro-batch.
   - Use the `num_items_in_batch` mechanism of transformers' Trainer, passing the summed weights instead of the token count.
   - With gradient accumulation and DDP, the denominator must be the global total across the accumulation steps and all ranks.
   - Without this, short sequences are silently up-weighted.

Required test: on a tiny Gemma-4-config model, this loss equals the model's built-in loss within 1e-4 when all weights are 1.

Allowed alternative: `loss.impl: liger` uses Liger's fused linear cross-entropy, but only if it supports Gemma 4's softcap **[verify]**. The default is `selective`.

### 7.4 Model and LoRA

- **Base weights.**
  - Load `model.train_base_path` (QAT bf16 parent) with `torch_dtype=bfloat16`.
  - With `model.quantization: nf4`, load through bitsandbytes 4-bit (`bnb_4bit_quant_type=nf4`, `bnb_4bit_compute_dtype=bfloat16`, double quant on) and run `prepare_model_for_kbit_training`.
  - Load only what text training needs. Whether the vision and audio towers can be skipped or freed in transformers 5.13.1 is **[verify]**. At minimum, they must stay frozen and carry no LoRA.
- **Tokenizer and template.** Load both from `model.served_path`, the `-qat-w4a16-ct` folder. Record the template's sha256. If `model.expected_template_sha256` is set and differs, that is a hard preflight failure.
- **LoRA targets.**
  - Use a regex: `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`.
  - For E4B, also exclude `k_proj`/`v_proj` on the KV-shared layers, which receive no gradient. Read the layer indices from the model config (`num_kv_shared_layers` or similar) **[verify name]**.
  - `bias: none`, `lora_dropout: 0.0`, `use_rslora: false`, and `init_lora_weights: true`.
- **Attention.** Use `attn_implementation: sdpa` by default.
- **Gradient checkpointing.** On, with `use_reentrant=False`.
- **Parallelism.** Set by `train.parallelism`:
  - `ddp` (E4B default): `accelerate launch --num_processes 4`. Each GPU holds a full copy of the model and trains on different data.
  - `model_split` (31B default): a single process with `device_map="auto"` and a `max_memory` cap per GPU (default 20 GiB, leaving room for activations). It is slow (one GPU busy at a time) but simple and robust.
  - `fsdp_qlora`: listed for later, **not built in v1**. Leave room for it in the code.

### 7.5 Batching and order

- **Batch shape.** `per_device_train_batch_size: 1` with gradient accumulation. Default `grad_accum: 16` for model_split, and `4` for DDP (4 GPUs × 4 = effective 16). LoRA prefers small effective batches (playbook §5).
- **Order.** Use a length-grouped random order, shuffled with the run seed. **The first `train.longest_first_steps` (default 3) optimizer steps use the longest examples**, so an out-of-memory error shows up in minute one, not hour seven.
- **Packing.** No packing. Our sequences are long, and varlen attention does not cover head dim 512.

### 7.6 Time-aware learning-rate schedule (WSD)

The schedule is a function of **elapsed time**, not only steps:

- **Warmup.** Linear warmup over `schedule.warmup_steps` (default 20 optimizer steps).
- **Stable.** Hold at peak `lr`.
- **Decay.**
  - The adapter's training ends at `t_end = min(deadline, projected time to finish max_epochs)`. Re-project it every step from an exponential moving average of step time.
  - The decay window starts at `t_end − decay_frac × train_duration`, with `decay_frac` defaulting to 0.15.
  - Over that window the LR falls linearly to `lr × schedule.final_lr_ratio` (default 0.1).
  - Once decay has started, freeze `t_end`, so the decay is not stretched or squeezed by noise in the step-time projection.
- **Stop.** Stop when `now ≥ t_end` or epochs are done. Then save the final adapter.

Implement this as a `LambdaLR` whose lambda reads a shared clock object. Test it by simulating clocks (§11).

### 7.7 Several adapters in one run

Adapters train **one after another** in config order, each in a fresh process. Time is split up front **[design call]**:

- Each adapter's share of the training window is `time_share` if set. Otherwise it is in proportion to `supervised_tokens × max_epochs`, using the counts from preflight.
- If an adapter finishes early (its epochs are done), the leftover time goes to the next adapter.
- A skipped or failed adapter's time also goes to the next one.

### 7.8 Checkpoints and resume

- **When.** Every `train.checkpoint_every_min` (default 45) wall-clock minutes, and at a clean stop.
- **Where and what.** Saved to `runs/<run_id>/checkpoints/<adapter>/step_<n>/`, containing:
  - the adapter;
  - optimizer, scheduler and RNG states;
  - the sampler position (epoch and index);
  - elapsed training time;
  - `schedule_state` (whether decay has started, and the frozen `t_end`).
- **Retention.** Keep the last `train.keep_checkpoints` (default 2) for each adapter. A 31B r=32 checkpoint with optimizer state is about 2–3 GB **[estimate]**, so watch the `/kaggle/working` size limit.
- **Resume.**
  - `run.resume_from: <path>` points to a previous run's output directory (attached as a dataset). The train phase then:
    1. loads the newest valid checkpoint for each adapter;
    2. skips adapters marked finished;
    3. continues with a **new** time budget.
  - If the decay had not started, resume continues the stable phase.
  - If it had started, resume finishes the decay quickly (over at most 10% of the new budget) and then **re-enters a stable phase** only if `resume.extend: true`. Otherwise it stops. **[design call]**

### 7.9 Signals logged during training

Write to `runs/<run_id>/metrics/<adapter>.jsonl` every `log_every_steps` (default 5):

- step, epoch, elapsed, LR;
- train loss, grad norm;
- tokens/s (all tokens and supervised tokens);
- supervised tokens in the step;
- peak memory per GPU.

Validation loss is computed every `val_every_min` (default 60) wall-clock minutes and at the end.

Two checks run during training:

- **Dead-adapter check.** After `dead_check_step` (default 50) steps, compute ‖B‖ for each LoRA module. Report modules that are still exactly zero, grouped by module type and layer range. It only warns; it does not abort.
- **Throughput projection.** After 10 steps, print the measured tokens/s, how many passes over the data the budget allows, and the projected end time. The same goes into the report.

### 7.10 HF-side parity reference

At the end of each adapter's training, still in the training process with the model loaded:

1. Take `parity.num_prompts` (default 8) fixed prompts from the validation set: an assistant turn's context, cut to ≤ 4k tokens.
2. Run greedy decoding for `parity.gen_tokens` (default 64) tokens, recording each chosen token's log-prob and the top-5 log-probs.
3. Do it twice: **with** the adapter, and **without** it (`disable_adapter()`).
4. Save the results to `runs/<run_id>/parity/<adapter>_hf.json`. The eval phase compares them with vLLM (§9.3).

---

## 8. Export and package

### 8.1 Export (per adapter)

- **Weights.** Load the final adapter weights (or a checkpoint, per §6.2) on CPU and cast them to **bf16**.
- **Key check.**
  - Every key must match `^base_model\.model\.model\.language_model\.layers\.\d+\..+\.lora_[AB]\.weight$`.
  - There must be no vision or audio keys.
  - The layer and module set must agree with `target_modules`.
  - Compare against the sample adapter's key format.
- **`adapter_config.json`.**
  - Set `base_model_name_or_path` to the **served** model id (e.g. `google/gemma-4-31b-it-qat-w4a16-ct`) and `inference_mode: true`.
  - Record `r` and `lora_alpha`.
  - Expand `target_modules` to an explicit list or keep the regex, whichever vLLM 0.19.1 accepts **[verify]**. Test by loading in the eval phase.
- **Sanity.**
  - Rank ≤ 128.
  - No NaN or Inf.
  - Not all B matrices are zero.
  - File sizes are reasonable.
- **Output** goes to `runs/<run_id>/adapters/<name>/`.

### 8.2 Package

1. Copy `submission.template_dir` to `runs/<run_id>/submission/`. The agent track owns this directory: `agent.yaml`, prompts, sub-agents, sampling and eval configs. The default for tests is `build/gemma-4-developer-agent/sample_submission/` without its adapters.
2. Set `model:` in `agent.yaml` and every `sub_agents/*.yaml` to `submission.served_model_id`. Edit the line as text so `!include` tags are kept exactly.
3. Fill `adapters/` according to `submission.adapter_map`, for example `{main_lora: main_lora, tool_lora: tool_lora}` or `{main_lora: shared, tool_lora: shared}`.
   - The keys are the names the YAMLs reference; the values are the trained adapters. Sharing is done by copying.
   - Every `adapter:` named in the YAMLs must exist in `adapters/`. Otherwise apply the §6.2 fallback order, and in the worst case remove that `adapter:` line and report it.
4. Validate. All of these must pass:
   - `pipeline.src.utils.packager.validate_submission_directory`;
   - `adk_submission.validate_directory`, the official validator also used in the starter notebook;
   - our own checks: extension whitelist, no symlinks, total size < 3 GiB, `max_output_tokens` and `thinking_budget` ≤ 32,768, adapter names match the regex.
5. Zip to `runs/<run_id>/submission.zip`, then copy it to `/kaggle/working/submission.zip`. Record its sha256 and size.

---

## 9. Eval in the final hour

### 9.1 Server

- Start `adk_submission.VllmServer(VllmConfig(...))` with the same settings as the starter notebook and the scorer:
  - `tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`, `max_model_len=32768`;
  - `gpu_memory_utilization=0.90`, `tensor_parallel_size=4`;
  - `enable_lora=True`, `max_loras=<number of adapters>`, `max_lora_rank=<max rank>`;
  - `startup_timeout=1200`.
- Serve the **staged submission directory**, i.e. exactly what is in the zip. The 12B model would also need `--hf-overrides` (see the starter). Copy the starter's code for this rather than re-deriving it.
- If the official helper offers it, prefer the same "compile submission → start server" path the scorer uses (`discover_adapters`, `create_model_registry`) **[verify]**.

### 9.2 Three-task smoke eval

- **Tasks.** `eval.task_ids` lists 3 fixed ids from `tasks.jsonl`. Choose easy-to-medium tasks from different repos; the Evaluation track or the owner picks them. Fixed ids make runs comparable. Preflight asserts these ids exist and are **not** in the training data.
- **How it runs.** Use `swegemma.evaluate.Evaluator(EvalConfig(...))` with `sandbox='subprocess'` (no Docker on Kaggle). Use the same `EventsCompactionConfig` as the scorer: `compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5`.
  - Task inputs: snapshots, graphs and embeddings come from the attached competition data.
  - Per-task limits come from the template's `eval_config.yaml`.
  - Run the 3 tasks concurrently if the Evaluator supports it **[verify]**, otherwise one after another.
- **Deadline.** The eval phase has a hard deadline. A task still running then counts as `timeout` (fail).
- **Output.** Print a block like this, and write `eval/score.json`:

```
================ SMOKE EVAL — 3 fixed tasks (NOT a leaderboard estimate) ================
model: gemma-4-31b-it-qat-w4a16-ct   adapters: main_lora(r32) tool_lora(r32)   run: 2026-11-03_31b_full
task                      result   patch_applied  turns  tool_call_errors  time
fastapi_15661             PASS     yes            23     0                 11m42s
<id 2>                    FAIL     yes            31     2                 14m05s
<id 3>                    TIMEOUT  no             —      —                 15m00s
SCORE: 1/3 = 33.3%   (95% Clopper-Pearson interval: 0.8%–90.6%)
parity: main_lora OK (greedy agreement 0.97 vs base 0.98) | tool_lora OK (0.95 vs 0.97)
==========================================================================================
```

Use the exact field names the swegemma results provide for pass, patch applied, turns and tool-call errors **[builder]**. If a field does not exist, show `—`.

### 9.3 Adapter parity check (runs first, before the tasks)

For each adapter, send the §7.10 prompts to the running vLLM server: `/v1/completions`, `temperature=0`, `max_tokens=parity.gen_tokens`, `logprobs=5`, with that adapter's model name. Then compare with the HF reference:

- **Greedy agreement**: the fraction of positions, up to the first divergence and across all prompts, where vLLM's token equals HF's.
- **Mean |Δ log-prob|** of the HF-chosen token, wherever both report it.
- Run the same comparison for the **base model without an adapter**. This is the baseline gap, caused by int4 versus bf16/NF4.

The parity result is `OK` if the adapter's agreement is within `parity.max_agreement_drop` (default 0.10) of the base agreement. Otherwise it is `SUSPECT`.

`SUSPECT` does **not** block packaging (the zip already exists). It shows prominently in the report with the recommendation: "retrain at r=16 or with fewer target modules (vllm#50059)". The first calibration run should train one r=16 and one r=32 E4B adapter, to settle D2 early.

### 9.4 What the 3-task score means

A model whose true pass rate is 30% scores 0/3 about a third of the time, so this score **cannot rank models**. It proves three things:

- the adapter loads;
- the agent runs end to end;
- the agent produces patches that apply.

Model comparisons use validation loss, the parity check and the Evaluation track's larger runs. The report says this in one line.

---

## 10. Configuration and launch

### 10.1 Config (example, `configs/31b_full.yaml`; `base.yaml` holds all defaults)

```yaml
run:
  name: 31b_full
  seed: 1234
  output_root: /kaggle/working/runs        # local: ./runs
  resume_from: null                        # path to a previous run dir (attached dataset)
  session_start_env: SESSION_T0            # epoch seconds, set by the notebook's first cell

budget: {session_hours: 12.0, safety_margin_min: 20, eval_min: 60, package_min: 5, export_min: 10,
         setup_max_min: 10, preflight_max_min: 15}

model:
  train_base_path: /kaggle/input/...gemma-4-31b-it-qat-q4_0-unquantized/...   # [verify] mount path
  served_path:     /kaggle/input/...gemma-4-31b-it-qat-w4a16-ct/...           # tokenizer + chat template
  served_model_id: gemma-4-31b-it-qat-w4a16-ct                                # written into agent YAMLs
  expected_template_sha256: null           # set once known; mismatch = hard fail
  quantization: nf4                        # bf16 | nf4
  attn_implementation: sdpa

data:
  train_files: []                          # REQUIRED; given per run, never hardcoded
  tasks_file: /kaggle/input/gemma-4-developer-agent/tasks.jsonl   # leakage check + eval
  successful_only: true
  val_fraction: 0.05
  val_max_examples: 64
  template_kwargs: {}                      # e.g. enable_thinking: true — must match how the harness renders [verify]
  cache_dir: null                          # optional pre-tokenized cache from an attached dataset

lora: {r: 32, alpha: 64, dropout: 0.0,
       target_regex: '.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$',
       exclude_kv_shared: true}

train:
  parallelism: model_split                 # ddp | model_split
  lr: 1.0e-4
  max_epochs: 2
  max_seq_length: 16384
  per_device_batch: 1
  grad_accum: 16
  weight_decay: 0.0
  max_grad_norm: 1.0
  longest_first_steps: 3
  checkpoint_every_min: 45
  keep_checkpoints: 2
  val_every_min: 60
  log_every_steps: 5
  dead_check_step: 50
  max_restarts: 2
  min_restart_min: 20
  oom_fallbacks: [{max_seq_length: 12288}, {max_seq_length: 8192}]

schedule: {warmup_steps: 20, decay_frac: 0.15, final_lr_ratio: 0.1}
loss: {impl: selective, chunk_rows: 4096, tool_call_weight: 1.0}

adapters:                                  # trained in this order
  - name: main_lora
    roles: [root]
    time_share: null                       # null = proportional to supervised tokens × epochs
    overrides: {}                          # any lora/train/loss key, e.g. {lora: {r: 16}}
  - name: tool_lora
    roles: [code_analyzer]
    overrides: {}

parity: {num_prompts: 8, gen_tokens: 64, max_agreement_drop: 0.10}

submission:
  template_dir: null                       # REQUIRED for real runs; agent track's submission dir
  adapter_map: {main_lora: main_lora, tool_lora: tool_lora}
  fallback_adapters_dir: null

eval:
  enabled: true
  task_ids: []                             # exactly 3 fixed ids from tasks.jsonl
  snapshots_dir: /kaggle/input/...         # [verify] mount layout of competition data
  vllm: {gpu_memory_utilization: 0.90, tensor_parallel_size: 4, max_model_len: 32768, startup_timeout: 1200}
```

**Rules for the config loader:**

- Unknown keys are an error.
- `--set a.b=c` overrides any key, with values parsed as YAML.
- A per-adapter `overrides` entry is deep-merged on top of the globals.
- The resolved config is written to the run directory before anything else happens.
- `run.mode: calibration` replaces the budget with `calibration_minutes` (default 30). It trains each adapter for a few minutes, then runs export, package and a 1-task eval. Use it to measure memory and speed and to test the whole path cheaply.

### 10.2 Kaggle notebook (`kaggle/multi_lora_sft.ipynb`)

| Cell | What it does |
|---|---|
| 1 | `os.environ["SESSION_T0"] = str(time.time())`. This must be the first line of the notebook. |
| 2 | Install the official wheelhouse, restoring the `+cu128` suffix as the starter does. Then install our private training wheelhouse with `--no-deps`. Print the versions. |
| 3 | `!python -m pipeline.multi_lora_sft.run --config <dataset path>/pipeline/multi_lora_sft/configs/31b_full.yaml --set data.train_files=[...] --set eval.task_ids=[...]` |
| 4 | Print `REPORT.md`. |

**Kernel settings.** Add a pipeline entry to `scripts/kaggle_settings.json` (the user approves and pushes; do not push yourself):

- `machine_shape: NvidiaL4`, `enable_internet: false`, the starter's `docker_image` pin, `is_private: true`;
- datasets: the repo dataset, the official wheelhouse, our training wheelhouse and our private data dataset;
- models: the served `-w4a16-ct` model and the training parent.

### 10.3 Run directory (`/kaggle/working/runs/<date>_<name>/`)

```
resolved_config.yaml   manifest.json   REPORT.md   submission.zip
status/<phase>.json    logs/<phase>.log            metrics/<adapter>.jsonl
preflight/report.json  (counts, length histograms, drop/mismatch counts, leakage result, projections)
cache/<adapter>/       checkpoints/<adapter>/step_<n>/   adapters/<name>/   parity/
submission/            (staged, unzipped)            eval/results/   eval/score.json
```

`manifest.json` holds:

- the run id and T0;
- a code hash (sha256 over `pipeline/**/*.py` and `*.yaml`, since Kaggle has no `.git`; also the `git_sha` that `kaggle_push.py` stamps, if present);
- sha256 of every data file, of the chat template and of the tokenizer;
- library versions, GPU names and seeds;
- per-adapter results: steps, epochs, tokens seen, final train/val loss, dead modules, parity verdict, adapter source used for packaging;
- every phase's planned and actual duration and state;
- the zip's sha256 and size, and the eval score.

---

## 11. Testing

All tests run on CPU with `uv run pytest pipeline/multi_lora_sft/tests`. Use a **tiny random Gemma-4-config model** (2 layers, small hidden size and vocab, softcap on, built from the real config class) so no download is needed **[builder]**.

| Area | Test |
|---|---|
| Config | Defaults load. Unknown keys fail. `--set` overrides work. Per-adapter override merges are correct. Invalid models, rank > 128 and bad adapter names fail. |
| Clock | Deadlines from T0 add up. A phase gets the right budget. Leftover time is redistributed between adapters. |
| Schedule | Simulated clocks: warmup, stable, decay. Decay starts at the right time when the deadline comes first, and when the epochs end first. `t_end` is frozen after decay starts. Resume mid-stable and mid-decay. |
| Loss | Equals the model's built-in loss (weights = 1) within 1e-4. Weighting changes the right tokens. Normalization stays correct under gradient accumulation (two micro-batches of different lengths). |
| Data | Role filtering. The split is by task_id and deterministic. Over-long trajectories are counted as dropped. Leakage detection catches a planted test task (by `task_id`/`instance_id`, by `(repo, base_commit)`, and by an identical gold patch). Cache key changes when the template changes. |
| LoRA targets | The regex matches language-model linears only, never the vision tower. The KV-shared exclusion works. |
| Export | Keys match the sample adapter's format. bf16. Config fields rewritten. A planted vision key fails. |
| Package | The sample template plus fake adapters produces a valid zip. The `model:` rewrite keeps `!include`. A missing adapter applies the fallback order. Size and extension violations are caught. |
| Orchestrator | Fake phases that fail, hang (killed at the deadline) or succeed: `package` and `report` always run, and the status and manifest are correct. |
| End to end | `--set run.mode=calibration` with the tiny model, `eval.enabled=false` and `data/train_trajectories.jsonl` runs every phase on CPU and produces a valid zip. |

### Leakage check definition

The **hard fail** cases are:

- a training `task_id` equals any `instance_id` in `tasks.jsonl`;
- a training `(repo, base_commit)` equals a test task's;
- the final `submit_patch` diff of a trajectory, normalized for whitespace, equals a test task's gold `patch`.

A **warning only**, listed in the preflight report, is a trajectory on the same repo as a test task within ±N commits **[builder]**.

---

## 12. Build order

Each step ends in something runnable.

1. **Environment.**
   - Find out which of `peft trl accelerate datasets` the pinned L4 image already has. A 10-minute L4 session, or reading the image's package list, will tell you.
   - Build the private training wheelhouse to match. Check that it imports together with the official wheelhouse on an L4 session **[verify]**.
2. **Skeleton.** Config, clock, run context, the orchestrator with fake phases, and the package phase (using the sample template). This already guarantees "always ship". Write its tests.
3. **Preflight.** Data contract, split, leakage check, tokenization via `processor`, cache, report.
4. **Train on E4B with DDP.** Model loading, LoRA regex, selective loss, WSD schedule, callbacks, checkpoint and resume. Then a CPU end-to-end run with the tiny model.
5. **Export and parity reference.**
6. **Eval.** VllmServer, parity check, 3-task smoke eval, score printout. Reuse the starter notebook's code.
7. **Report and notebook.** Add the `kaggle_settings.json` entry for the user to approve.
8. **First L4 session: E4B calibration (30 min).** Measure memory, tokens/s, step time and vLLM start time. Test r=16 against r=32 parity. Update §5's budgets and the playbook's §7 estimates with the real numbers.
9. **31B model_split** on the calibration path, then the first full run.

---

## 13. Risks

| Risk | Effect | Mitigation in this design |
|---|---|---|
| vLLM LoRA on an int4 base misbehaves at r=32 (vllm#50059) | The adapter scores worse when served than its training suggests | Parity check (§9.3); r=16 fallback; test at calibration |
| Template or rendering differs from what the harness feeds the model (reasoning re-sent in history? thinking flag?) | The model learns a context it never sees | Served template by path and sha; `template_kwargs`; **[verify]** against a real `traces/trace_<id>.json` before the first full run |
| Not enough GPU memory at 16k context | Run dies in the first minutes | Longest-first steps; `oom_fallbacks`; calibration run |
| 31B is compute-bound (~600 trajectory passes in 10 h, split over 2 adapters) | Undertrained adapters | Prefer quality over quantity in data; consider `shared` for 31B; FSDP-QLoRA later |
| Training libraries are incompatible with `transformers==5.13.1` | Nothing runs | Build step 1 checks this first |
| Session killed before the zip exists | No submission | Package runs right after export, eval runs last, and 20 min margin |
| Over-long trajectories dropped | Model is biased away from hard tasks | Dropped fraction reported; windowing in v2 |
| Only one run per week | Slow iteration | CPU preflight, calibration runs, resume |

---

## 14. Questions only the owner can answer

These do not block starting the build (steps 1–7). Answer them before the first full run.

1. **D5: TRL instead of Unsloth.** Is that acceptable?
2. **D10: tool-call loss weight.** Should it default to 1.0, with 1.5 as an A/B config, or should 1.5 be the default as in the original plan?
3. **The 3 eval task ids.** Who picks them, the owner or the Evaluation track?
4. **Submission template.** Which directory is the agent track's real template, and where does it live in the repo or a dataset?
5. **Quota.** Is about one full L4 run per week right, or is there more GPU time to plan for?

## 15. Left to the builder

- Module and function names, and the internal structure inside each phase.
- The Arrow or Parquet cache format and how many tokenization workers to use.
- Exactly how the tiny test model is built.
- How to find swegemma's per-task result fields (pass, patch applied, turns, tool-call errors).
- Whether to run the 3 eval tasks concurrently.
- Whether to use Liger, if it is shown to be correct and faster.

Every **[verify]** item above becomes a fact once checked. Update this doc and the playbook when it is.
