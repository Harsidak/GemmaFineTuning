"""Picks the trainer backend (PyTorch/TRL or JAX/Tunix) from the model and the hardware.

One notebook, one switch: `train.backend = auto` means

    E4B / E2B on a TPU VM  ->  tunix   (JAX, tpu_training/)
    everything else        ->  trl     (PyTorch; 31B QLoRA model-split on 4xL4, E4B DDP on GPUs, CPU smoke runs)

Only environment markers are inspected here (no jax / torch import), so the orchestrator can decide before it
spawns the training subprocess, and so this module imports instantly in any environment.
"""

from __future__ import annotations

import os
from pathlib import Path

TUNIX_MODEL_MARKERS = ("e4b", "e2b")


def tpu_present(env: dict[str, str] | None = None, dev_root: str | Path = "/dev") -> bool:
    e = os.environ if env is None else env
    if e.get("TPU_NAME") or e.get("TPU_ACCELERATOR_TYPE") or e.get("COLAB_TPU_ADDR") or e.get("KAGGLE_TPU"):
        return True
    try:
        return any(Path(dev_root).glob("accel*")) or any(Path(dev_root).glob("vfio/[0-9]*"))
    except OSError:
        return False


def is_small_gemma(served_model_id: str) -> bool:
    sid = served_model_id.lower()
    return any(marker in sid for marker in TUNIX_MODEL_MARKERS)


def resolve_backend(requested: str, *, served_model_id: str, accelerator: str = "auto", tpu: bool | None = None) -> str:
    """'trl' or 'tunix'. An explicit request always wins; `auto` follows the rule in the module docstring."""
    if requested in {"trl", "tunix"}:
        return requested
    has_tpu = tpu_present() if tpu is None else tpu
    wants_tpu = accelerator == "tpu" or (accelerator == "auto" and has_tpu)
    return "tunix" if (wants_tpu and is_small_gemma(served_model_id)) else "trl"
