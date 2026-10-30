"""Tunix (JAX) training backend for Gemma 4 E4B / E2B on TPU.

Everything specific to Tunix lives in this folder; the pipeline only calls `tpu_training.phase.run_tunix_adapter_phase`
when `train.backend` resolves to `tunix`. Resume, task ledger, guards, best-checkpoint selection, tracking, the adapter
contract check and the cross-session bundle are NOT reimplemented here: they come from
`pipeline.multi_lora_sft.shared`, exactly as for the PyTorch backend.

Modules
-------
runtime        JAX platform detection and the device mesh
hf_config      Hugging Face `config.json` -> Tunix `ModelConfig`
lora_convert   qwix LoRA tensors <-> PEFT `adapter_model.safetensors` (pure numpy, exact)
model_io       load the base model, attach LoRA, functional forward, chunked loss
batching       static-shape batches from the shared Parquet token cache
trainer        the training loop (WSD schedule, grad accumulation, checkpoints, eval)
phase          `run_tunix_adapter_phase`: same status / summary contract as `pipeline.multi_lora_sft.phases.train`
probe          `python -m tpu_training.probe`: a few-minute TPU readiness check
"""
