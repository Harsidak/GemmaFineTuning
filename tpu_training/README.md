# `tpu_training/` — Tunix (JAX) backend for Gemma 4 E4B / E2B on TPU

Everything Tunix-specific lives here. The pipeline calls `tpu_training.phase.run_tunix_adapter_phase` only when
`train.backend` resolves to `tunix`; nothing else in the repo imports JAX.

```
train.backend = auto   ->  tunix  for E4B/E2B on a TPU (accelerator tpu, or TPU_* env / /dev/accel*)
                           trl    for everything else (31B, any GPU, CPU)
train.backend = trl | tunix  forces one (validated up front)
```

## What is shared with the PyTorch backend (not copied)

`pipeline/multi_lora_sft/shared/` is torch-free and jax-free and is used by both backends through one object,
`TrainingSession`:

| Concern | Where |
| :--- | :--- |
| resume discovery + compatibility (model / LoRA shape fatal, optimizer reuse decided by backend) | `shared/session.py`, `shared/compat.py` |
| task ledger — which training problems were consumed (`progress.json`), resume at problem 4,174 | `shared/ledger.py` |
| atomic, verified, pruned checkpoints; disk guard; best-adapter copy | `shared/checkpoints.py` |
| NaN/Inf and loss-divergence guards, early stopping, best-vs-last selection | `shared/guards.py`, `session.py` |
| W&B policy (off while tuning, offline without key/network) | `shared/tracking.py` |
| adapter contract check against the served w4a16 checkpoint | `shared/adapter_contract.py` |
| cross-session bundle (`lora_checkpoints/`) + secrets scan | `shared/bundle.py`, `shared/secrets_scan.py` |
| Optuna (`python -m pipeline.multi_lora_sft.tune`) | `shared/tuning.py`, `tune.py` |

Only the math is here:

| Module | Role |
| :--- | :--- |
| `runtime.py` | JAX platform detection, `('fsdp','tp')` mesh |
| `hf_config.py` | Hugging Face `config.json` → Tunix `ModelConfig` (E2B/E4B/31B presets, derived fields otherwise), KV-sharing layout check |
| `model_io.py` | load HF safetensors, attach qwix LoRA, `train_hidden` forward (hidden states, no `[B,T,262144]` logits), chunked softcapped cross-entropy, `set_remat` |
| `lora_convert.py` | qwix LoRA tensors ⇄ PEFT `adapter_model.safetensors`, exact, pure numpy |
| `peft_io.py` | adapter dir writer/reader (`adapter_config.json`, `matched_target_modules.json`), optimizer-state files |
| `batching.py` | static-shape batches from the shared Parquet cache |
| `trainer.py` | jitted grad / accumulate / AdamW update; f32 master LoRA, bf16 compute; non-finite steps are skipped on-device |
| `loop.py` | WSD schedule, sampler + ledger, guards, wall-clock checkpoints, validation, stop handling |
| `phase.py` | `run_tunix_adapter_phase`: the `train:<adapter>` phase contract (status JSON, manifest summary, `final_adapter/`) |
| `probe.py` | `python -m tpu_training.probe`: load → parity → LoRA steps → export → contract, with timings |
| `tools/` | Tunix ⇄ Hugging Face+PEFT logit parity check (two scripts, two environments) |

## Design points worth knowing

* **LoRA must be attached with remat off.** qwix creates LoRA parameters lazily during its first call; inside
  `nnx.remat` they are discarded. `attach_lora` runs first, `set_remat("BLOCK")` afterwards.
* **The training forward is a registered method** (`train_hidden`). qwix applies LoRA by intercepting `einsum` while a
  registered method runs; calling the layers from outside would silently train nothing (grad norm 0).
* **Same multiplier as serving.** qwix scale = `alpha/r`; PEFT/vLLM scale = `alpha/r` (`alpha/sqrt(r)` with rsLoRA).
  `qwix_alpha()` converts so training and serving agree.
* **Fused k/v.** Tunix's `kv_einsum` shares one A matrix between k and v; the exporter writes the same A twice. An
  adapter trained by the PyTorch backend (independent A matrices) therefore cannot be loaded into Tunix: the phase
  logs the reason, starts a fresh LoRA and does **not** skip any training data.
* **Checkpoints** are the shared layout (`step_NNNNNN/{trainer_resume.json, adapter/, optimizer_state.safetensors,
  progress.json}`); the optimizer state is only reused when the checkpoint came from the same backend.
* **Fixed shapes.** `train.pad_to_multiple_of` (default 512 on TPU) bounds the number of XLA compilations; the first step
  of every new length is slow.
* `lora.dropout` and `lora.target_regex` are not applied (LoRA covers q/k/v/o and gate/up/down of the language model);
  the phase records a note.

## Verified vs. not verified

Verified (JAX-CPU, tiny Gemma 4 checkpoint, `tests/`):

* gradient of the chunked loss against finite differences; B=0 adapter equals the base model; chunked loss equals dense loss
* fresh run, wall-clock checkpoints, best-adapter selection, resume from the ledger with optimizer restore, damaged
  checkpoint fallback, NaN abort without checkpointing
* an adapter written by this package, loaded with Hugging Face + PEFT, reproduces Tunix's logits to 1.7e-6
* the whole suite also passes in an environment holding only the packages in `kaggle/build_train_wheels.py::TPU_TESTED`

**Not verified: anything on a real TPU** (HBM use and tokens/s of the real E4B, XLA compile times, the Kaggle TPU image's
Python/libtpu). Run `python -m tpu_training.probe` (or `RUN["probe"]=True` in the notebook) first.

## Running the tests

```bash
# needs jax, flax, qwix, google-tunix (see requirements-tpu.txt) and a tiny checkpoint:
python - <<'EOF'
from pipeline.multi_lora_sft.training.model import save_tiny_gemma4_checkpoint   # needs torch + transformers
save_tiny_gemma4_checkpoint("fixture", vocab_size=512, num_hidden_layers=4, num_kv_shared_layers=2, include_vision=True, seed=42)
EOF
TUNIX_TINY_FIXTURE=fixture/<created dir> JAX_PLATFORMS=cpu python -m pytest tpu_training/tests
```

`tests/test_lora_convert.py` is pure numpy and runs everywhere; `test_tunix_integration.py` is skipped without JAX.

## Wheels

`kaggle/build_train_wheels.py` builds two wheelhouses in one notebook run: `train_wheels/` (PyTorch stack) and
`train_wheels_tpu/` (JAX + Tunix + libtpu; `google-tunix` is installed `--no-deps` because it declares tensorflow-datasets,
numba, … that the trainer never imports). The training notebook installs the TPU wheelhouse only for the Tunix backend and
falls back to pip when the notebook has internet. `WHEEL_BUNDLES=tpu WHEEL_PYTHON=3.13` rebuilds only the TPU bundle for
another Python version.
