"""Torch-free checkpoint publishing, listing and verification (identical for every backend).

Layout (kept stable so TRL, Tunix, the orchestrator and the cross-session bundle all read the same thing):

    <root>/step_<NNNNNN>/trainer_resume.json
                         adapter/adapter_model.safetensors   (+ adapter_config.json, PEFT format)
                         progress.json                        (task-level ledger)
                         ...backend-specific state (optimizer, rng, scheduler)
    <root>/best_adapter/adapter_model.safetensors             (+ best.json: step, val_loss)
"""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .fsio import atomic_write_json, dir_size_bytes, free_gib, publish_dir, safe_rmtree

logger = logging.getLogger(__name__)

STEP_RE = re.compile(r"^step_(\d+)$")
ADAPTER_FILE = "adapter_model.safetensors"
RESUME_FILE = "trainer_resume.json"
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class CheckpointError(RuntimeError):
    """A checkpoint could not be written or failed post-write verification."""


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Parses a .safetensors header without loading tensors and checks that every tensor fits in the file."""
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("file shorter than the 8-byte safetensors prefix")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > 100 * 1024 * 1024 or 8 + n > size:
            raise ValueError(f"implausible safetensors header length {n} for a {size}-byte file")
        header = json.loads(fh.read(n).decode("utf-8"))
    data_len = size - 8 - n
    tensors: dict[str, Any] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if not (0 <= start <= end <= data_len):
            raise ValueError(f"tensor {name!r} offsets {start}:{end} exceed the {data_len}-byte data section (truncated file?)")
        width = _DTYPE_BYTES.get(info["dtype"])
        if width is not None:
            elems = 1
            for d in info["shape"]:
                elems *= int(d)
            if elems * width != end - start:
                raise ValueError(f"tensor {name!r} shape {info['shape']} does not match its {end - start}-byte span")
        tensors[name] = info
    return {"tensors": tensors, "metadata": header.get("__metadata__") or {}}


def list_checkpoints(root: str | Path) -> list[Path]:
    """Complete `step_*` checkpoints sorted oldest -> newest by global step."""
    r = Path(root)
    if not r.exists():
        return []
    found = [
        d
        for d in r.iterdir()
        if STEP_RE.match(d.name)
        and d.is_dir()
        and (d / RESUME_FILE).is_file()
        and (d / "adapter" / ADAPTER_FILE).is_file()
    ]
    return sorted(found, key=lambda d: int(STEP_RE.match(d.name).group(1)))  # type: ignore[union-attr]


def find_latest_checkpoint(root: str | Path) -> Path | None:
    found = list_checkpoints(root)
    return found[-1] if found else None


def checkpoint_is_loadable(ckpt: str | Path) -> bool:
    """Cheap, torch-free integrity check: resume metadata parses and the adapter header is internally consistent."""
    c = Path(ckpt)
    try:
        meta = json.loads((c / RESUME_FILE).read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or "step" not in meta:
            raise ValueError("trainer_resume.json has no 'step'")
        hdr = read_safetensors_header(c / "adapter" / ADAPTER_FILE)
        if not hdr["tensors"]:
            raise ValueError("adapter has 0 tensors")
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "do not resume from this one"
        logger.warning("Checkpoint %s is unreadable (%s); skipping it", c, exc)
        return False


def write_checkpoint(
    root: str | Path,
    step: int,
    write_fn: Callable[[Path], None],
    *,
    keep: int = 2,
    min_free_disk_gib: float = 0.0,
    fallback_expected_bytes: int = 0,
    protect: Iterable[Path] = (),
    reason: str = "wallclock",
) -> Path | None:
    """Writes `step_<n>/` atomically: temp dir -> `write_fn(tmp)` -> verify -> rename -> prune.

    Disk safety: the expected size is the previous checkpoint's size (else `fallback_expected_bytes`). If free space
    would drop below `min_free_disk_gib`, older checkpoints beyond the newest are pruned first; if it is still short the
    save is skipped (returns None) instead of crashing training with ENOSPC. A write that fails verification raises
    `CheckpointError` and leaves previous checkpoints untouched.
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    protected = {Path(p).resolve() for p in protect}
    existing = list_checkpoints(root_p)
    expected = dir_size_bytes(existing[-1]) if existing else int(fallback_expected_bytes)
    floor = float(min_free_disk_gib)
    if free_gib(root_p) - expected / 1024**3 < floor:
        for old in existing[:-1]:
            if old.resolve() not in protected:
                safe_rmtree(old)
        if free_gib(root_p) - expected / 1024**3 < floor:
            logger.warning(
                "Skipping %s checkpoint at step %d: %.2f GiB free, need ~%.2f GiB + %.1f GiB floor",
                reason, step, free_gib(root_p), expected / 1024**3, floor,
            )
            return None

    final = root_p / f"step_{int(step):06d}"
    tmp = root_p / f".tmp_step_{int(step):06d}"
    safe_rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_fn(tmp)
        if not checkpoint_is_loadable(tmp):
            raise CheckpointError(f"checkpoint written to {tmp} failed verification")
    except BaseException:
        safe_rmtree(tmp)
        raise
    publish_dir(tmp, final)

    existing = list_checkpoints(root_p)
    while len(existing) > max(1, int(keep)):
        victim = existing.pop(0)
        if victim.resolve() in protected:
            continue
        safe_rmtree(victim)
    return final


def write_best_adapter(root: str | Path, step: int, val_loss: float, copy_fn: Callable[[Path], None]) -> Path:
    """Publishes `<root>/best_adapter/` (adapter weights + best.json) atomically."""
    root_p = Path(root)
    tmp = root_p / ".tmp_best_adapter"
    safe_rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        copy_fn(tmp)
        if not (tmp / ADAPTER_FILE).is_file():
            raise CheckpointError("best-adapter writer did not produce adapter_model.safetensors")
        read_safetensors_header(tmp / ADAPTER_FILE)
        atomic_write_json(tmp / "best.json", {"step": int(step), "val_loss": float(val_loss), "saved_at_ts": time.time()})
    except BaseException:
        safe_rmtree(tmp)
        raise
    return publish_dir(tmp, root_p / "best_adapter")


def read_best_adapter(root: str | Path) -> tuple[Path, dict[str, Any]] | None:
    d = Path(root) / "best_adapter"
    try:
        meta = json.loads((d / "best.json").read_text(encoding="utf-8"))
        read_safetensors_header(d / ADAPTER_FILE)
        return d, meta
    except Exception:  # noqa: BLE001
        return None
