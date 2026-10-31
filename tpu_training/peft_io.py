"""Adapter and optimizer-state files for the JAX trainer.

Adapters are written in exactly the layout the PyTorch backend writes (`adapter_model.safetensors` with PEFT keys,
`adapter_config.json`, `matched_target_modules.json`), so the export phase, the adapter-contract check, the bundle
and a later PyTorch resume read them without knowing which backend produced them. Tensors stay float32 here; the
export phase casts to bf16.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .lora_convert import PEFT_PREFIX, ConversionError, qwix_to_peft

logger = logging.getLogger(__name__)

ADAPTER_FILE = "adapter_model.safetensors"
OPT_FILE = "optimizer_state.safetensors"
_MODULE_RE = re.compile(rf"^{re.escape(PEFT_PREFIX)}\.(\d+)\.(.+)\.lora_[AB]\.weight$")


def effective_scale(r: int, alpha: float, use_rslora: bool) -> float:
    """The multiplier PEFT / vLLM apply to B @ A at serve time."""
    return float(alpha) / (math.sqrt(r) if use_rslora else r)


def qwix_alpha(r: int, alpha: float, use_rslora: bool) -> float:
    """The `alpha` to give qwix (scale = alpha / r) so training uses the same multiplier serving will."""
    return effective_scale(r, alpha, use_rslora) * r


def matched_modules(peft_tensors: dict[str, Any]) -> list[str]:
    mods = set()
    for k in peft_tensors:
        m = _MODULE_RE.match(k)
        if m:
            mods.add(f"model.language_model.layers.{m.group(1)}.{m.group(2)}")
    return sorted(mods)


def write_adapter_dir(
    out_dir: str | Path,
    lora_np: dict[str, np.ndarray],
    *,
    r: int,
    alpha: float,
    use_rslora: bool = False,
    base_model: str = "",
    target_regex: str | None = None,
) -> dict[str, np.ndarray]:
    """qwix LoRA arrays -> `out_dir/{adapter_model.safetensors, adapter_config.json, matched_target_modules.json}`."""
    from safetensors.numpy import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    peft = qwix_to_peft(lora_np)
    for k, v in peft.items():
        if not np.isfinite(v).all():
            raise ConversionError(f"refusing to write non-finite tensor {k!r}")
    mods = matched_modules(peft)
    save_file(peft, str(out / ADAPTER_FILE), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model, "inference_mode": False,
        "r": int(r), "lora_alpha": float(alpha), "lora_dropout": 0.0, "bias": "none", "use_rslora": bool(use_rslora),
        "target_modules": mods,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "matched_target_modules.json").write_text(
        json.dumps({"target_regex": target_regex, "matched_modules": mods}, indent=2), encoding="utf-8"
    )
    return peft


def read_adapter_tensors(adapter_dir: str | Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return {k: np.asarray(v, dtype=np.float32) for k, v in load_file(str(Path(adapter_dir) / ADAPTER_FILE)).items()}


# ------------------------------------------------------------------------------------------ optimizer state
def save_opt_state(path: str | Path, leaves: list[np.ndarray]) -> None:
    """Optimizer leaves in tree order. Position + shape is the key; a layout change is detected on load."""
    from safetensors.numpy import save_file

    save_file({f"leaf_{i:05d}": np.ascontiguousarray(a) for i, a in enumerate(leaves)}, str(path), metadata={"n": str(len(leaves))})


def load_opt_state(path: str | Path, expected_shapes: list[tuple[int, ...]]) -> list[np.ndarray] | None:
    """Leaves for an optimizer whose fresh state has `expected_shapes`; None (with a log line) when they do not match."""
    from safetensors.numpy import load_file

    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = load_file(str(p))
    except Exception as exc:  # noqa: BLE001 - truncated / corrupt file
        logger.warning("optimizer state %s unreadable (%s); starting with fresh Adam moments", p, exc)
        return None
    leaves = [data.get(f"leaf_{i:05d}") for i in range(len(expected_shapes))]
    if len(data) != len(expected_shapes) or any(a is None for a in leaves):
        logger.warning("optimizer state %s has %d leaves, expected %d; starting with fresh Adam moments", p, len(data), len(expected_shapes))
        return None
    for i, (a, shp) in enumerate(zip(leaves, expected_shapes)):
        if tuple(a.shape) != tuple(shp):
            logger.warning("optimizer state leaf %d shape %s != %s; starting with fresh Adam moments", i, a.shape, shp)
            return None
    return leaves  # type: ignore[return-value]
