# Multi-LoRA SFT Pipeline (`pipeline.multi_lora_sft`)

Time-budgeted, fault-tolerant multi-LoRA supervised fine-tuning, export, ADK submission packaging, and vLLM smoke evaluation for the **Gemma 4 Developer Agent** competition (`4 × NVIDIA L4`, 12-hour offline Kaggle sessions).

See [`design_doc.md`](./design_doc.md) for architecture, memory math, WSD schedule design, and failure matrix.

## Phases

Each phase runs in an isolated subprocess under a hard wall-clock deadline from `SESSION_T0`:

1. **`setup`** — optional offline wheel installation, core import check, GPU & input path verification.
2. **`preflight`** — trajectory schema check, 3-way benchmark leakage gate (`task_id`, `(repo, base_commit)`, normalized gold diff), served tokenizer/template verification, and Parquet tokenization cache.
3. **`train:<adapter>`** — sequential per-adapter LoRA/QLoRA training with selective-logits softcapped cross-entropy, longest-first length-grouped sampling, time-aware WSD LR decay, OOM fallback retries, wall-clock checkpointing/resume, and HF greedy parity reference generation.
4. **`export`** — CPU `bfloat16` cast, strict `language_model`-only safetensors key & rank sanity checks, and `adapter_config.json` rewrite targeting the served `-qat-w4a16-ct` model.
5. **`package`** *(always runs)* — stages `agent.yaml` + `sub_agents/`, resolves `submission.adapter_map` with 4-tier fallback down to bare base model, runs all 3 submission validators, and builds `submission.zip`.
6. **`eval`** — starts `VllmServer`, runs greedy parity against HF references (`vllm#50059` guard), and executes a 3-task `swegemma` smoke evaluation.
7. **`report`** *(always runs)* — writes `manifest.json` and `REPORT.md`.

## Hardware

| `train.accelerator` | `train.parallelism` | Use for |
| :--- | :--- | :--- |
| `cuda` | `model_split` | 31B NF4 QLoRA split across 4×L4 (`configs/31b_full.yaml`) |
| `cuda` | `ddp` | E4B bf16 LoRA, one replica per GPU (`configs/e4b_full.yaml`) |
| `tpu` | `spmd` | E4B bf16 LoRA sharded over all TPU chips with torch_xla FSDPv2 (`configs/e4b_tpu.yaml`) |
| `tpu` / `cuda` / `cpu` | `single` | one device (TPU single core, one GPU, or CPU smoke tests) |
| `auto` | any | picks TPU > CUDA > CPU at runtime and remaps CUDA-only parallelism on TPU |

TPU specifics: no bitsandbytes (bf16 only), sequences padded to `train.pad_to_multiple_of` (default 512) and the
`static` loss so XLA compiles a bounded set of graphs, the SPMD batch is global (multiple of the chip count),
parity generation and the vLLM smoke eval are skipped. The TPU path has unit coverage for config and loss
equivalence but has not yet been run on real TPU hardware: start with `run.mode=calibration`.

## Robustness

* **Model won't load**: setup validates both checkpoints (config.json, every shard in the index, non-empty weights),
  rejects a compressed-tensors `-w4a16-ct` checkpoint as a training base, and fails if train/served sizes differ.
  Loads are offline (`run.hf_offline`); `device_map` offload to CPU/disk is reported as OOM so fallbacks apply.
* **OOM**: `train.oom_fallbacks` retries (only the latest attempt's log is inspected); TPU `RESOURCE_EXHAUSTED` counts.
* **NaN/Inf**: `train.max_nonfinite_logs` consecutive bad logs abort to the last good checkpoint (never saved).
* **Deadline / SIGTERM**: train phases get `train.sigterm_grace_sec` to stop at a step boundary and save the adapter.
* **Crash mid-run**: an emergency checkpoint is written (`train.emergency_checkpoint`) before the restart.
* **Checkpoints**: atomic, global step numbering across resumes, disk-aware (`train.min_free_disk_gib`), unreadable
  newest checkpoints are skipped on resume, already-finished checkpoints are exported without retraining.
* **Served agent**: `submission.generation` writes `configs/sampling.yaml` and `submission.eval_budget` writes
  `eval_config.yaml` (the sample template's 1-minute / 10-call budget is never shipped).

## Notebook

`kaggle/multi_lora_sft.ipynb` exposes every knob: pick a `preset` (`auto`, `smoke_cpu`, `e4b_gpu_calibration`,
`e4b_gpu_full`, `e4b_tpu`, `31b_gpu_full`), edit `PARAMS` (`None` = keep default), optional `ADAPTER_OVERRIDES`.
It auto-adjusts to the detected hardware/disk/session limits, validates the full config in-process, prints the
effective per-adapter table, and only then launches the orchestrator (`RUN["dry_run"]=True` stops after validation).

## CLI Usage

```bash
# Local CPU calibration smoke run (tiny Gemma 4 fixture, no downloads required)
uv run python -m pipeline.multi_lora_sft.run \
  --config pipeline/multi_lora_sft/configs/e4b_calibration.yaml \
  --set run.output_root=./runs/local_smoke \
  --set model.train_base_path=__tiny__ \
  --set model.served_path=__tiny__ \
  --set data.train_files='[data/train_trajectories.jsonl]' \
  --set data.tasks_file=build/gemma-4-developer-agent/tasks.jsonl \
  --set submission.working_zip_path=null \
  --set eval.enabled=false

# Via unified PipelineRunner CLI
uv run python pipeline/main.py --action multi_lora_sft \
  --config pipeline/multi_lora_sft/configs/31b_full.yaml \
  --set data.train_files='[/kaggle/input/.../train.jsonl]'

# Run CPU unit + end-to-end test suite
uv run pytest pipeline/multi_lora_sft/tests
```
