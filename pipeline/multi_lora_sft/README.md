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
