"""JAX platform detection and the device mesh for the Tunix backend."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AcceleratorUnavailable(RuntimeError):
    """A TPU was demanded but JAX sees none (or a platform this backend does not support was requested)."""


def resolve_platform(requested: str) -> str:
    """'tpu' or 'cpu' for a `train.accelerator` value, from what JAX can actually see.

    `auto` prefers a TPU; `tpu` fails loudly (a silent CPU fallback would burn the whole session on a 30x slower run);
    `cuda` is the PyTorch backend's job.
    """
    import jax

    if requested == "cuda":
        raise AcceleratorUnavailable("train.accelerator='cuda' belongs to the PyTorch backend (train.backend='trl')")
    platforms = {d.platform for d in jax.devices()}
    if requested == "tpu":
        if "tpu" not in platforms:
            raise AcceleratorUnavailable(
                f"train.accelerator='tpu' but JAX sees {sorted(platforms)}. Is the notebook's accelerator set to a TPU, "
                "and are jax[tpu] / libtpu in the attached wheel bundle (kaggle/build_train_wheels.py --tpu)?"
            )
        return "tpu"
    if requested == "cpu":
        return "cpu"
    return "tpu" if "tpu" in platforms else "cpu"


def device_info() -> dict[str, Any]:
    import jax

    devs = jax.devices()
    return {
        "platform": devs[0].platform if devs else None,
        "count": len(devs),
        "kinds": sorted({getattr(d, "device_kind", "?") for d in devs}),
        "process_count": jax.process_count(),
        "jax": jax.__version__,
    }


def make_mesh(parallelism: str) -> Any:
    """Mesh with the axis names Tunix's Gemma sharding rules use: ('fsdp', 'tp').

    `spmd` shards parameters and the batch over every device (fully-sharded data parallel); `single` uses one device.
    Tensor parallelism stays 1: E4B fits sharded over the FSDP axis and TP would only add communication.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    devices = jax.devices()
    if parallelism == "single":
        devices = devices[:1]
    return Mesh(np.array(devices).reshape(len(devices), 1), ("fsdp", "tp"))
