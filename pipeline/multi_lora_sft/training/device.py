"""Accelerator abstraction for CUDA, TPU (`torch_xla`), and CPU training.

Everything device-specific that the training loop needs lives here so the rest of the pipeline can stay
backend-agnostic:
- `resolve_accelerator()` maps `train.accelerator: auto|cuda|tpu|cpu` to the backend actually present and fails
  fast with an actionable message when an explicitly requested backend is missing.
- RNG capture/restore, peak-memory stats, CPU-materialised state dicts, and `torch.save` wrappers that work on
  XLA tensors (which must be synced with `mark_step()` and copied to host before serialisation).
- `is_oom_text()` recognises CUDA, ROCm, TPU (`RESOURCE_EXHAUSTED`) and CPU allocator OOM signatures.

`torch_xla` is imported lazily; nothing here requires it unless a TPU is actually used.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACCELERATORS: frozenset[str] = frozenset({"auto", "cuda", "tpu", "cpu"})

OOM_SIGNATURES: tuple[str, ...] = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "hip out of memory",
    "resource_exhausted",
    "ran out of memory in memory space hbm",
    "defaultcpuallocator: can't allocate memory",
    "insufficient accelerator memory",
)


class AcceleratorUnavailableError(RuntimeError):
    """Raised when the configured accelerator cannot be used in this environment."""


def is_oom_text(text: str | None) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(sig in low for sig in OOM_SIGNATURES)


def xla_installed() -> bool:
    return importlib.util.find_spec("torch_xla") is not None


def _xla_device_type() -> str | None:
    """Returns the PJRT device type (`"TPU"`, `"CUDA"`, `"CPU"`) or None when torch_xla is unusable."""
    if not xla_installed():
        return None
    try:
        import torch_xla.runtime as xr

        return str(xr.device_type())
    except Exception as exc:  # noqa: BLE001 - torch_xla raises many runtime-specific errors
        logger.debug("torch_xla present but runtime unavailable: %s", exc)
        return None


def tpu_available() -> bool:
    if not xla_installed():
        return False
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or os.environ.get("TPU_NAME") or os.environ.get(
        "TPU_ACCELERATOR_TYPE"
    ):
        return True
    return _xla_device_type() == "TPU"


def resolve_accelerator(requested: str = "auto") -> str:
    """Resolves `auto` to `tpu` > `cuda` > `cpu`, validating explicit requests."""
    import torch

    req = (requested or "auto").lower()
    if req not in ACCELERATORS:
        raise AcceleratorUnavailableError(f"Unknown accelerator {requested!r}; expected one of {sorted(ACCELERATORS)}")

    if req == "tpu":
        if not xla_installed():
            raise AcceleratorUnavailableError(
                "train.accelerator='tpu' but torch_xla is not installed. On a Kaggle/GCP TPU VM install the "
                "torch_xla wheel matching your torch version (e.g. `pip install torch_xla[tpu]`), or set "
                "train.accelerator=auto to fall back to CUDA/CPU."
            )
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        return "tpu"
    if req == "cuda":
        if not torch.cuda.is_available():
            raise AcceleratorUnavailableError(
                "train.accelerator='cuda' but torch.cuda.is_available() is False (no GPU attached, or a CPU-only "
                "torch build is installed)."
            )
        return "cuda"
    if req == "cpu":
        return "cpu"

    if tpu_available():
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        return "tpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def accelerator_inventory() -> dict[str, Any]:
    """Describes every accelerator visible to this process (used by setup + manifest)."""
    import torch

    inv: dict[str, Any] = {"cuda": [], "tpu": None, "xla_installed": xla_installed()}
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            inv["cuda"].append(
                {"index": idx, "name": props.name, "total_memory_gib": round(props.total_memory / (1024**3), 2)}
            )
    if tpu_available():
        try:
            import torch_xla.runtime as xr

            inv["tpu"] = {
                "device_type": str(xr.device_type()),
                "global_device_count": int(xr.global_runtime_device_count()),
                "accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE"),
            }
        except Exception as exc:  # noqa: BLE001
            inv["tpu"] = {"error": str(exc)}
    return inv


def xla_device_count() -> int:
    import torch_xla.runtime as xr

    return int(xr.global_runtime_device_count())


def enable_spmd() -> None:
    """Switches torch_xla into SPMD mode; must run before any XLA tensor is created."""
    import torch_xla.runtime as xr

    xr.use_spmd()


def mark_step(accelerator: str) -> None:
    if accelerator == "tpu":
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def capture_rng_state(accelerator: str) -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if accelerator == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if accelerator == "tpu":
        try:
            import torch_xla.core.xla_model as xm

            state["torch_xla"] = int(xm.get_rng_state())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not capture XLA RNG state: %s", exc)
    return state


def restore_rng_state(saved: dict[str, Any], accelerator: str) -> None:
    import random

    import numpy as np
    import torch

    if "python" in saved:
        random.setstate(saved["python"])
    if "numpy" in saved:
        np.random.set_state(saved["numpy"])
    if "torch_cpu" in saved:
        torch.set_rng_state(saved["torch_cpu"])
    if accelerator == "cuda" and torch.cuda.is_available() and "torch_cuda" in saved:
        try:
            torch.cuda.set_rng_state_all(saved["torch_cuda"])
        except (RuntimeError, ValueError, TypeError) as exc:
            # Different GPU count than when the checkpoint was written; data order is still restored.
            logger.warning("Could not restore CUDA RNG state (%s); continuing with fresh CUDA RNG", exc)
    if accelerator == "tpu" and "torch_xla" in saved:
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(int(saved["torch_xla"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore XLA RNG state: %s", exc)


def peak_memory_mb(accelerator: str) -> list[float]:
    import torch

    if accelerator == "cuda" and torch.cuda.is_available():
        return [round(torch.cuda.max_memory_allocated(i) / (1024 * 1024), 1) for i in range(torch.cuda.device_count())]
    if accelerator == "tpu":
        try:
            import torch_xla.core.xla_model as xm

            info = xm.get_memory_info(xm.xla_device())
            used = info.get("bytes_used") or (info.get("kb_total", 0) - info.get("kb_free", 0)) * 1024
            return [round(float(used) / (1024 * 1024), 1)]
        except Exception:  # noqa: BLE001
            return []
    return []


def reset_peak_memory(accelerator: str) -> None:
    import torch

    if accelerator == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def empty_cache(accelerator: str) -> None:
    import gc

    import torch

    gc.collect()
    if accelerator == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def cpu_state_dict(state: dict[str, Any], accelerator: str) -> dict[str, Any]:
    """Detaches and copies every tensor to host memory (required before saving XLA tensors)."""
    import torch

    mark_step(accelerator)
    out: dict[str, Any] = {}
    for key, val in state.items():
        out[key] = val.detach().to("cpu").contiguous() if torch.is_tensor(val) else val
    return out


def save_object(obj: Any, path: Path, accelerator: str) -> None:
    """`torch.save` that also works for objects holding XLA tensors (optimizer / scheduler state)."""
    import torch

    if accelerator == "tpu":
        import torch_xla.core.xla_model as xm

        xm.save(obj, str(path), master_only=True, global_master=True)
        return
    torch.save(obj, path)
