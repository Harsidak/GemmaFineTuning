"""The JAX math of one LoRA SFT run: jitted gradient / accumulate / update / eval, optimizer-state I/O.

Numerics: the frozen base model runs in `compute_dtype` (bf16 on TPU); the trainable LoRA tensors are kept as float32
MASTER copies and cast to `compute_dtype` inside the jitted loss, so gradients flow back to float32 and AdamW updates
do not lose small steps to bf16 rounding. The loss is a SUM of weighted token NLLs; the optimizer step divides the
accumulated gradient by the TOTAL supervised weight of the whole optimizer step, so short sequences are never
up-weighted (same normalisation as the PyTorch backend). A non-finite gradient leaves weights and moments untouched.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import model_io
from .batching import place

logger = logging.getLogger(__name__)


class LoraTrainer:
    def __init__(
        self,
        lora_model: Any,
        *,
        mesh: Any | None,
        compute_dtype: Any,
        chunk_rows: int,
        weight_decay: float,
        max_grad_norm: float,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ) -> None:
        import jax
        import jax.numpy as jnp
        import optax
        from flax import nnx
        from tunix.sft import utils as su

        self.mesh, self.compute_dtype, self.chunk_rows = mesh, compute_dtype, int(chunk_rows)
        self.model = lora_model
        gd, lora, rest = model_io.split_model(lora_model)
        self.graphdef, self.rest = gd, rest
        self.master = jax.tree.map(lambda x: x.astype(jnp.float32), lora)
        self._template = model_io.lora_template(lora)
        self.n_params = int(sum(int(np.prod(s)) for s in self._template.values()))

        parts = [optax.clip_by_global_norm(float(max_grad_norm))] if max_grad_norm and max_grad_norm > 0 else []
        parts.append(optax.scale_by_adam(b1=adam_betas[0], b2=adam_betas[1], eps=adam_eps))
        if weight_decay:
            parts.append(optax.add_decayed_weights(float(weight_decay)))
        self.tx = optax.chain(*parts)
        self.opt_state = self.tx.init(self.master)

        cdt, chunk = compute_dtype, self.chunk_rows

        def loss_sum(master: Any, rest_: Any, tokens: Any, mask: Any, labels: Any, weights: Any) -> tuple[Any, Any]:
            lora_c = jax.tree.map(lambda x: x.astype(cdt), master)
            m = nnx.merge(gd, lora_c, rest_)
            attn = su.make_causal_attn_mask(mask)
            pos = su.build_positions_from_mask(mask)
            h = model_io.hidden_states(m, tokens, pos, attn)
            return model_io.chunked_cross_entropy(m, h, labels, weights, chunk_rows=chunk)

        self._grad = jax.jit(jax.value_and_grad(loss_sum, has_aux=True))
        self._eval = jax.jit(loss_sum)
        self._add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b), donate_argnums=0)
        self._zeros = jax.jit(lambda t: jax.tree.map(jnp.zeros_like, t))
        tx = self.tx

        def apply(master: Any, opt_state: Any, acc: Any, w_total: Any, lr: Any) -> tuple[Any, Any, Any, Any]:
            g = jax.tree.map(lambda x: x / jnp.maximum(w_total, 1.0), acc)
            gnorm = optax.global_norm(g)
            updates, new_opt = tx.update(g, opt_state, master)
            new_master = jax.tree.map(lambda p, u: p - lr * u, master, updates)
            ok = jnp.isfinite(gnorm)
            keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(ok, a, b), new, old)
            return keep(new_master, master), keep(new_opt, opt_state), gnorm, ok

        self._apply = jax.jit(apply, donate_argnums=(0, 1, 2))

    # ------------------------------------------------------------------ one optimizer step
    def step(self, micro_batches: list[dict[str, np.ndarray]], lr: float) -> dict[str, Any]:
        """Accumulate gradients over `micro_batches`, then apply one AdamW update at learning rate `lr`."""
        import jax.numpy as jnp

        if not micro_batches:
            raise ValueError("optimizer step with zero micro-batches")
        acc = self._zeros(self.master)
        nll_total, w_total = jnp.float32(0.0), jnp.float32(0.0)
        for mb in micro_batches:
            d = place(mb, self.mesh)
            (nll, ws), g = self._grad(self.master, self.rest, d["tokens"], d["mask"], d["labels"], d["weights"])
            acc = self._add(acc, g)
            nll_total, w_total = nll_total + nll, w_total + ws
        self.master, self.opt_state, gnorm, ok = self._apply(
            self.master, self.opt_state, acc, w_total, jnp.float32(lr)
        )
        wt = float(w_total)
        return {
            "loss": float(nll_total) / max(wt, 1.0),
            "grad_norm": float(gnorm),
            "applied": bool(ok),
            "weight_sum": wt,
        }

    def eval_loss(self, batches: list[dict[str, np.ndarray]]) -> float | None:
        nll_sum = w_sum = 0.0
        for b in batches:
            d = place(b, self.mesh)
            nll, ws = self._eval(self.master, self.rest, d["tokens"], d["mask"], d["labels"], d["weights"])
            nll_sum, w_sum = nll_sum + float(nll), w_sum + float(ws)
        return nll_sum / w_sum if w_sum > 0 else None

    # ------------------------------------------------------------------ state <-> host
    def lora_numpy(self) -> dict[str, np.ndarray]:
        return model_io.lora_to_numpy(self.master)

    def load_lora_numpy(self, arrays: dict[str, np.ndarray]) -> None:
        self.master = model_io.numpy_to_lora(self.master, arrays)

    @property
    def template(self) -> dict[str, tuple[int, ...]]:
        return self._template

    def opt_leaves(self) -> list[np.ndarray]:
        import jax

        return [np.asarray(jax.device_get(x)) for x in jax.tree.leaves(self.opt_state)]

    def opt_shapes(self) -> list[tuple[int, ...]]:
        import jax

        return [tuple(x.shape) for x in jax.tree.leaves(self.opt_state)]

    def load_opt_leaves(self, leaves: list[np.ndarray]) -> None:
        import jax

        flat, treedef = jax.tree.flatten(self.opt_state)
        new = [
            jax.device_put(np.asarray(a, dtype=x.dtype), getattr(x, "sharding", None)) if getattr(x, "sharding", None) is not None
            else jax.numpy.asarray(a, dtype=x.dtype)
            for a, x in zip(leaves, flat)
        ]
        self.opt_state = jax.tree.unflatten(treedef, new)

    def dead_lora_report(self) -> dict[str, Any]:
        """Same idea as the PyTorch check: LoRA B tensors whose norm is still exactly zero after training began."""
        arrs = self.lora_numpy()
        b = {k: v for k, v in arrs.items() if k.endswith("_lora_b")}
        dead = sorted(k for k, v in b.items() if float(np.linalg.norm(v)) == 0.0)
        return {"total_lora_b_modules": len(b), "dead_lora_b_modules": len(dead), "dead_grouped": dead[:8]}
