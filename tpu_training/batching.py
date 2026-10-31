"""Static-shape batches for the JAX trainer (numpy in, device arrays out).

XLA compiles one program per input shape, so batches have a fixed batch dimension (short batches are padded with
fully-masked rows) and a sequence length rounded up to `pad_multiple`; the number of distinct shapes is then at most
`max_seq_length / pad_multiple`. The cached Parquet rows are the ones the PyTorch backend trains on: `labels[t]` is the
token AT position t (or IGNORE_INDEX), so the target for the hidden state at t is `labels[t + 1]` and the batcher shifts
labels and weights by one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

IGNORE_INDEX = -100


class BatchError(ValueError):
    pass


def round_up(n: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return int(n)
    return int(math.ceil(n / multiple) * multiple)


def collate(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    pad_id: int,
    pad_multiple: int | None = None,
    min_len: int = 2,
) -> dict[str, np.ndarray]:
    """Rows -> {'tokens', 'mask', 'labels', 'weights'}; labels/weights already shifted to next-token targets."""
    if not rows:
        raise BatchError("cannot build a batch from zero rows")
    if len(rows) > batch_size:
        raise BatchError(f"{len(rows)} rows do not fit a batch of {batch_size}")
    longest = max(len(r["input_ids"]) for r in rows)
    t = round_up(max(longest, min_len), pad_multiple)

    tokens = np.full((batch_size, t), int(pad_id), dtype=np.int32)
    mask = np.zeros((batch_size, t), dtype=bool)
    labels = np.full((batch_size, t), IGNORE_INDEX, dtype=np.int32)
    weights = np.zeros((batch_size, t), dtype=np.float32)
    for i, r in enumerate(rows):
        ids, lab = r["input_ids"], r["labels"]
        n = len(ids)
        if len(lab) != n:
            raise BatchError(f"row {i}: {n} input_ids but {len(lab)} labels")
        tokens[i, :n] = np.asarray(ids, dtype=np.int32)
        mask[i, :n] = True
        if n > 1:
            tgt = np.asarray(lab[1:], dtype=np.int32)
            labels[i, : n - 1] = tgt
            w = r.get("weights")
            wv = np.asarray(w[1:], dtype=np.float32) if w is not None else np.ones(n - 1, dtype=np.float32)
            weights[i, : n - 1] = np.where(tgt != IGNORE_INDEX, wv, 0.0)
    mask[len(rows):, 0] = True  # padding rows keep one visible token: an all-masked row would softmax over nothing
    return {"tokens": tokens, "mask": mask, "labels": labels, "weights": weights}


def batch_stats(batch: dict[str, np.ndarray]) -> dict[str, int]:
    return {"tokens": int(batch["mask"].sum()), "supervised": int((batch["labels"] != IGNORE_INDEX).sum())}


def take_micro_batches(order: Sequence[int], rows: Sequence[dict[str, Any]], *, batch_size: int, count: int) -> tuple[list[list[int]], list[int]]:
    """Split the next `batch_size * count` indices of `order` into micro-batches; returns (micro_batches, leftover)."""
    need = batch_size * count
    head, rest = list(order[:need]), list(order[need:])
    return [head[i : i + batch_size] for i in range(0, len(head), batch_size)], rest


def place(batch: dict[str, np.ndarray], mesh: Any | None) -> dict[str, Any]:
    """Host arrays -> device arrays, sharded over the mesh's 'fsdp' axis when the batch divides evenly."""
    import jax

    if mesh is None:
        return {k: jax.numpy.asarray(v) for k, v in batch.items()}
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    n_dev = int(mesh.shape.get("fsdp", 1))
    bsz = next(iter(batch.values())).shape[0]
    spec = P("fsdp", None) if n_dev > 1 and bsz % n_dev == 0 else P()
    sharding = NamedSharding(mesh, spec)
    return {k: jax.device_put(v, sharding) for k, v in batch.items()}
