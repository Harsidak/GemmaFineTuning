# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Training/eval/packaging workspace for the Kaggle **Gemma 4 Developer Agent** competition: build a SWE-bench-style coding agent on Gemma 4, scored by Pass@1 (patch applied to a clean clone, hidden pytest must pass; no partial credit). The deliverable is `submission.zip` — a **declarative** ADK agent (YAML + Markdown prompts + optional LoRA `.safetensors`), never Python entrypoints.

## Commands

Python 3.12, managed with `uv` (Windows dev machine; training/inference actually run on Kaggle GPUs).

```bash
uv sync                                                    # install deps (+ dev group: pytest, ruff, ipykernel)
uv run python pipeline/main.py --action status             # workspace readiness check
uv run python pipeline/main.py --action train --mode sft --config pipeline/configs/train_lora_sft.yaml
uv run python pipeline/main.py --action train --mode dpo   # defaults to configs/train_dpo_rl.yaml
uv run python pipeline/main.py --action evaluate --tasks data/val_tasks.jsonl
uv run python pipeline/main.py --action package --model <model> --adapter-path <dir>
uv run ruff check pipeline
```

`pipeline/scripts/{train,evaluate,package_submission}.py` are thin standalone CLIs over the same modules. There is no test suite yet (pytest is installed but no tests exist).

Real end-to-end evaluation uses the official `swegemma` harness (not installed in `.venv`; offline wheels live in `build/gemma-4-developer-agent/wheels/`):

```bash
swegemma eval --tasks tasks.jsonl --snapshots-dir snapshots --submission-dir <dir> \
  --results-dir results/run_01 --sandbox docker --task-id <id>
```

## Architecture

- `pipeline/main.py` — `PipelineRunner` is the single control plane (`status/train/evaluate/package`), used both by the CLI and by `kaggle/*.ipynb` launchers. It inserts the project root on `sys.path`, so all imports are absolute `pipeline.src.…`. On Kaggle, `pipeline/` is uploaded as a Dataset and imported from the notebooks — keep it importable without the rest of the repo.
- Packaging flow: `src/agent/builder.py` (`ADKSubmissionBuilder`) scaffolds `build/submission_staging/` (agent.yaml, prompts, sub_agents, sampling/eval configs, adapters) → `src/utils/packager.py` `validate_submission_directory` (root `agent.yaml`, no root `*.py`, single base model across root + `sub_agents/*.yaml`, supports `!include`) → zip to `submission.zip`.
- Training: `src/data` (JSONL loaders, trajectory→Gemma chat formatting) → `src/models` (QLoRA loader, Gemma LoRA target modules) → `src/training/{sft,dpo}_trainer.py` (TRL), driven by `pipeline/configs/*.yaml`.
- `src/agent/tools.py` `OFFICIAL_HOST_TOOLS` mirrors the harness's 9 tools; `src/utils/vllm_server.py` builds vLLM launch commands.
- **`evaluate` is currently a stub**: `PipelineRunner.evaluate` marks every task `passed: False` and only computes Pass@1 bookkeeping via `LocalEvaluator`. Real scores come from `swegemma eval`.

## Source of truth for competition rules

`build/gemma-4-developer-agent/HARNESS_README.md` (official harness docs) and `build/gemma-4-developer-agent/sample_submission/` are authoritative. `.agents/rules.md` and `README.md` are older summaries and disagree with the harness in places:

- **Allowed submission models** are only the quantized `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`. The pipeline's default `gemma-4-12b-it` is CLI-eval-only and would be rejected at Kaggle scoring. All agents must declare the same single model.
- **Tools** are the 9 in `tools.py`: `run_command, submit_patch, get_status, read_file, edit_file, write_file, search_similar_code, get_code_neighbors, get_code_subgraph`. `code_graph_search` (used in `pipeline/configs/agent_defaults.yaml` and README) does not exist in the harness.
- Root agent YAML schema follows the sample: `model:` string, `adapter:`, `instruction: !include …`, `generate_content_config: !include configs/sampling.yaml`, sub-agents via `agent_tool: {config_path, skip_summarization: true}`.
- Limits: unpacked submission < 3 GiB; allowed extensions `.yaml/.yml/.md/.txt/.py(skills only)/.json/.safetensors` (no `.bin/.pt`); `max_output_tokens` and `thinking_budget` ≤ 32,768 (vLLM `max_model_len` on 4×L4).
- Agent prompt gotchas: `submit_patch` diffs everything untracked in `/workspace` (put scratch scripts in `/tmp`), never touch the harness-written `/workspace/pytest.ini` / `conftest.py`, keep `edit_file` payloads small to avoid truncated tool calls.

## Research workflow conventions

- Every experiment gets `experiments/exp_<ID>_<name>/` (config.yaml + notes.md per `experiments/template.{yaml,md}`) with hypothesis and kill criteria written before running, and a row in `experiments/tracker.md`.
- Failures/hangs go in `planning/MISTAKES.md`; milestones in `planning/ROADMAP.md` (both gitignored, local only, as is `.agents/`).
- Papers/blogs/repos are cataloged in `resources/INDEX.md` with `_TEMPLATE.meta.yaml` sidecars; the actual files under `resources/{papers,web,repos}/` are gitignored.
- `writing/paper_outline.md` is the companion paper; claims there should be grounded in `experiments/tracker.md`.
- Never commit weights (`*.safetensors`, `*.bin`, …), `submission.zip`, `runs/`, or `logs/`.
