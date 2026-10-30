"""Tunix model plumbing: load the base weights, attach LoRA (qwix), split into trainable / frozen state, run the forward
pass up to the final norm, and compute the loss in chunks.

Why a custom forward + loss: `Gemma4.__call__` materialises logits for every position, i.e. `[B, T, 262144]` in float32
(about 8.6 GB for ONE 8k-token sequence). Training needs the hidden states, then a chunked, checkpointed
`softcap(h @ E^T)` cross-entropy that never holds more than `chunk_rows x vocab` logits. The forward below mirrors
`Gemma4.__call__` (tunix 0.1.7) up to `final_norm`; `check_forward_parity` proves at startup that it still equals the
library's own logits, so a Tunix upgrade that changes the model cannot silently change what we train.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LORA_MODULE_PATH = r".*(q_einsum|kv_einsum|k_einsum|attn_vec_einsum|gate_proj|up_proj|down_proj)"
IGNORE_INDEX = -100


class ModelLoadError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- loading
def choose_loader_mode(model_dir: str | Path) -> str:
    """'opt' (mmap fast path) only where it is safe: POSIX and one dtype per shard; else Tunix's per-tensor loader."""
    if os.name == "nt":
        return "orig"
    from pipeline.multi_lora_sft.shared.checkpoints import read_safetensors_header

    for f in sorted(Path(model_dir).glob("*.safetensors")):
        dtypes = {t["dtype"] for t in read_safetensors_header(f)["tensors"].values()}
        if len(dtypes) > 1:
            logger.warning("%s mixes dtypes %s; using the per-tensor loader", f.name, sorted(dtypes))
            return "orig"
    return "auto"


def load_base_model(model_dir: str | Path, model_cfg: Any, *, mesh: Any, dtype: Any) -> Any:
    from tunix.models.gemma4 import params_safetensors as ps

    d = str(model_dir)
    if not list(Path(d).glob("*.safetensors")):
        raise ModelLoadError(f"no .safetensors files in {d}")
    try:
        return ps.create_model_from_safe_tensors(d, model_cfg, mesh=mesh, dtype=dtype, mode=choose_loader_mode(d))
    except (ValueError, RuntimeError, AssertionError) as exc:
        raise ModelLoadError(f"loading {d} into Tunix failed: {exc}") from exc


def attach_lora(model: Any, *, rank: int, alpha: float, seed: int, dropout: float = 0.0, module_path: str = LORA_MODULE_PATH) -> Any:
    """qwix LoRA on the attention and MLP projections. A: he_uniform, B: zeros (so step 0 equals the base model)."""
    import qwix
    from flax import nnx

    if dropout:
        logger.warning("lora.dropout=%s is not applied by the Tunix backend (qwix dropout needs an rng per step)", dropout)
    provider = qwix.LoraProvider(module_path=module_path, rank=int(rank), alpha=float(alpha))
    lora_model = qwix.apply_lora_to_model(model, provider, rngs=nnx.Rngs(int(seed)), **model.get_model_input())
    n = sum(1 for _ in _lora_leaves(lora_model))
    if n == 0:
        raise ModelLoadError(f"qwix attached no LoRA tensors (module_path={module_path!r}); is this a Gemma 4 Tunix model?")
    logger.info("LoRA attached: %d tensors (rank %d, alpha %s)", n, rank, alpha)
    return lora_model


def _lora_leaves(model: Any):
    from flax import nnx

    state = nnx.state(model, nnx.LoRAParam)
    import jax

    return jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state))[0]


def split_model(model: Any) -> tuple[Any, Any, Any]:
    """(graphdef, trainable LoRA state, frozen state). The frozen state is never differentiated."""
    from flax import nnx

    return nnx.split(model, nnx.LoRAParam, ...)


def _path_key(path: Any) -> str:
    return "/".join(str(getattr(p, "key", p)) for p in path)


def lora_to_numpy(lora_state: Any) -> dict[str, np.ndarray]:
    """LoRA state -> {'layers/0/attn/q_einsum/w_lora_a': float32 ndarray}; host copies, safe to serialise."""
    import jax
    from flax import nnx

    flat = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(lora_state))[0]
    return {_path_key(p): np.asarray(jax.device_get(v), dtype=np.float32) for p, v in flat}


def lora_template(lora_state: Any) -> dict[str, tuple[int, ...]]:
    import jax
    from flax import nnx

    flat = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(lora_state))[0]
    return {_path_key(p): tuple(v.shape) for p, v in flat}


def numpy_to_lora(lora_state: Any, arrays: dict[str, np.ndarray]) -> Any:
    """A new LoRA state with `arrays` (same keys as `lora_to_numpy`) written in, keeping each leaf's sharding."""
    import jax
    from flax import nnx

    pure = nnx.to_pure_dict(lora_state)

    def put(path: Any, leaf: Any) -> Any:
        key = _path_key(path)
        if key not in arrays:
            raise ModelLoadError(f"LoRA tensor {key!r} missing from the restored adapter")
        arr = np.asarray(arrays[key], dtype=np.float32)
        if tuple(arr.shape) != tuple(leaf.shape):
            raise ModelLoadError(f"LoRA tensor {key!r}: restored shape {arr.shape} != model {tuple(leaf.shape)}")
        sharding = getattr(leaf, "sharding", None)
        return jax.device_put(arr.astype(leaf.dtype), sharding) if sharding is not None else jax.numpy.asarray(arr, leaf.dtype)

    restored = jax.tree_util.tree_map_with_path(put, pure)
    nnx.replace_by_pure_dict(lora_state, restored)  # in place: the State keeps its structure and variable metadata
    return lora_state


def cast_tree(tree: Any, dtype: Any) -> Any:
    import jax

    return jax.tree.map(lambda x: x.astype(dtype), tree)


# ----------------------------------------------------------------------------------------------- forward + loss
def hidden_states(model: Any, tokens: Any, positions: Any, attn_mask: Any) -> Any:
    """Final-norm hidden states `[B, T, D]`. Mirrors tunix 0.1.7 `Gemma4.__call__` for the no-cache training path."""
    x = model.embedder.encode(tokens)
    per_layer_inputs = None
    if model.config.per_layer_input_dim > 0:
        per_layer_inputs = model.embedder.encode_per_layer_input(x, tokens)
    transient_kvs: dict[str, Any] = {}
    for i, layer in enumerate(model.layers):
        shared_idx = model.kv_cache_sharing_patterns[i]
        kv_shared = None
        if shared_idx != i:
            shared_k, shared_v = transient_kvs[f"layer_{shared_idx}"]
            kv_shared = {"k": shared_k, "v": shared_v}
        _, x, kvs = layer(
            x, positions, None, attn_mask,
            per_layer_input=per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None,
            kv_shared_cache=kv_shared,
        )
        if i in model.shared_layer_origins:
            transient_kvs[f"layer_{i}"] = kvs
    return model.final_norm(x)


def logits_from_hidden(model: Any, x: Any) -> Any:
    import jax.numpy as jnp

    logits = model.embedder.decode(x).astype(jnp.float32)
    cap = model.config.final_logit_softcap
    if cap is not None:
        logits = jnp.tanh(logits / cap) * cap
    return logits


def chunked_cross_entropy(model: Any, hidden: Any, labels: Any, weights: Any, *, chunk_rows: int) -> tuple[Any, Any]:
    """(sum of weight * nll, sum of weight) over positions whose label is supervised.

    `labels[b, t]` is the token to predict at position t (already shifted by the batcher); IGNORE_INDEX positions get
    weight 0. Rows are processed `chunk_rows` at a time under `jax.checkpoint`, so peak memory is one chunk of
    logits, in forward and backward. All shapes are static (pad to a multiple of chunk_rows), which XLA requires.
    """
    import jax
    import jax.numpy as jnp

    b, t, d = hidden.shape
    n = b * t
    chunk = max(1, min(int(chunk_rows), n))
    pad = (-n) % chunk
    h = jnp.pad(hidden.reshape(n, d), ((0, pad), (0, 0)))
    y = jnp.pad(labels.reshape(n), (0, pad), constant_values=IGNORE_INDEX)
    w = jnp.pad(weights.reshape(n), (0, pad))
    h, y, w = (a.reshape(-1, chunk, *a.shape[1:]) for a in (h, y, w))

    @jax.checkpoint
    def one(carry: Any, xs: Any) -> tuple[Any, None]:
        hc, yc, wc = xs
        logits = logits_from_hidden(model, hc)
        valid = (yc != IGNORE_INDEX)
        safe = jnp.where(valid, yc, 0)
        lse = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, safe[:, None], axis=-1)[:, 0]
        wt = jnp.where(valid, wc, 0.0)
        return (carry[0] + jnp.sum((lse - picked) * wt), carry[1] + jnp.sum(wt)), None

    (nll, wsum), _ = jax.lax.scan(one, (jnp.float32(0.0), jnp.float32(0.0)), (h, y, w))
    return nll, wsum


def check_forward_parity(model: Any, *, atol: float = 2e-3) -> float:
    """Our hidden-state path + `logits_from_hidden` must equal the library's `model(...)` logits. Returns max |diff|."""
    import jax.numpy as jnp
    from tunix.sft import utils as su

    vocab = model.config.num_embed
    toks = (jnp.arange(16, dtype=jnp.int32)[None, :] * 7 + 3) % vocab
    mask = jnp.ones((1, 16), dtype=bool).at[:, -3:].set(False)
    attn, pos = su.make_causal_attn_mask(mask), su.build_positions_from_mask(mask)
    ref, _ = model(toks, pos, None, attn)
    ours = logits_from_hidden(model, hidden_states(model, toks, pos, attn))
    diff = float(jnp.max(jnp.abs((ref - ours)[:, :13])))
    if not np.isfinite(diff) or diff > atol:
        raise ModelLoadError(
            f"training forward disagrees with Tunix Gemma4.__call__ (max |diff| = {diff:.4g} > {atol}); "
            "the installed tunix version changed the model - pin tunix to the tested version"
        )
    return diff
