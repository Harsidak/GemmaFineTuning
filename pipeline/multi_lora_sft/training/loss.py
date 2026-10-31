"""Selective-logits cross-entropy with Gemma 4 softcapping, token weights, and SFTTrainer integration.

Implements design_doc.md §7.3:
1. Extracts final norm'd hidden states `h` of shape `[B, T, H]` before `lm_head`, avoiding
   materializing full `[B, T, 262144]` logits (saves ~8.6–17.2 GB per 16k sequence).
2. Shifts by 1 position and selects only supervised rows (`labels != -100`), shape `[N, H]`.
3. Processes `h_sel` in chunks of `chunk_rows` (default 4096):
   - computes `logits = lm_head(h_sel_chunk)` in fp32;
   - applies `softcap * tanh(logits / softcap)` from `config.get_text_config().final_logit_softcapping`;
   - computes per-token cross-entropy.
4. Normalizes by `Σ w_i` across the *entire optimizer step* (all gradient accumulation micro-batches
   and all DDP ranks) via Trainer's `num_items_in_batch` path so short sequences are never up-weighted.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ...src.data.processor import IGNORE_INDEX

logger = logging.getLogger(__name__)


def get_model_softcap(model: Any) -> float | None:
    """Reads `final_logit_softcapping` from a Gemma 4 (or PEFT/DDP-wrapped) model config."""
    cur = model
    for attr in ("module", "base_model", "model"):
        if hasattr(cur, "config"):
            break
        cur = getattr(cur, attr, cur)
    cfg = getattr(cur, "config", None)
    if cfg is None:
        return None
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    cap = getattr(text_cfg, "final_logit_softcapping", None)
    return float(cap) if cap is not None else None


def unwrap_causal_lm(model: Any) -> Any:
    """Unwraps DDP / Accelerate / PeftModel wrappers down to `Gemma4ForConditionalGeneration`."""
    cur = model
    if hasattr(cur, "module"):
        cur = cur.module
    if hasattr(cur, "get_base_model"):
        cur = cur.get_base_model()
    return cur


def _chunk_weighted_ce(
    lm_head: nn.Module,
    softcap: float | None,
    h_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    w_chunk: torch.Tensor | None,
) -> torch.Tensor:
    """`Σ w * CE` for one chunk of rows; the [rows, vocab] fp32 logits live only inside this call."""
    logits = lm_head(h_chunk).float()
    if softcap is not None and softcap > 0.0:
        logits = softcap * torch.tanh(logits / softcap)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_chunk.reshape(-1), reduction="none")
    if w_chunk is not None:
        ce = ce * w_chunk.reshape(-1).to(ce.device)
    return ce.sum()


def _run_chunk(
    lm_head: nn.Module,
    softcap: float | None,
    h_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    w_chunk: torch.Tensor | None,
    recompute: bool,
) -> torch.Tensor:
    # Without recompute, autograd keeps every chunk's [rows, 262144] fp32 logits until backward
    # (~1 GiB per 1k supervised tokens); with it only the [rows, H] hidden chunk is kept.
    if recompute and torch.is_grad_enabled() and h_chunk.requires_grad:
        return torch_checkpoint(
            _chunk_weighted_ce, lm_head, softcap, h_chunk, y_chunk, w_chunk, use_reentrant=False
        )
    return _chunk_weighted_ce(lm_head, softcap, h_chunk, y_chunk, w_chunk)


def _normalise(weighted_sum: torch.Tensor, weight_sum: torch.Tensor | float | None, fallback: torch.Tensor) -> torch.Tensor:
    if weight_sum is not None:
        denom = (
            weight_sum.to(device=weighted_sum.device, dtype=torch.float32)
            if torch.is_tensor(weight_sum)
            else torch.tensor(float(weight_sum), device=weighted_sum.device, dtype=torch.float32)
        )
    else:
        denom = fallback.to(device=weighted_sum.device, dtype=torch.float32)
    return weighted_sum / torch.clamp(denom, min=1e-8)


def static_cross_entropy_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    softcap: float | None = None,
    chunk_rows: int = 4096,
    weight_sum: torch.Tensor | float | None = None,
    ignore_index: int = IGNORE_INDEX,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Same objective as `selective_cross_entropy_loss` but with only static shapes (no boolean indexing,
    no `.item()` host syncs), so XLA/TPU compiles one graph per padded sequence length.

    Unsupervised positions are computed and zero-weighted instead of being gathered out, so it is a little
    more compute than the selective path on GPU but avoids a recompilation for every distinct token count.
    """
    if hidden_states.size(1) <= 1:
        return hidden_states.sum() * 0.0

    h_shift = hidden_states[:, :-1, :]
    y_shift = labels[:, 1:]
    mask = y_shift.ne(ignore_index)
    w = mask.to(torch.float32)
    if weights is not None:
        w = w * weights[:, 1:].to(torch.float32)
    y_safe = torch.where(mask, y_shift, torch.zeros_like(y_shift))

    head_device = next(lm_head.parameters()).device
    bsz, seq = y_shift.shape
    tokens_per_chunk = max(1, int(chunk_rows) // max(1, bsz))

    weighted_sum: torch.Tensor | None = None
    for start in range(0, seq, tokens_per_chunk):
        end = min(seq, start + tokens_per_chunk)
        part = _run_chunk(
            lm_head,
            softcap,
            h_shift[:, start:end].to(head_device),
            y_safe[:, start:end].to(head_device),
            w[:, start:end].to(head_device),
            checkpoint_chunks,
        )
        weighted_sum = part if weighted_sum is None else weighted_sum + part

    assert weighted_sum is not None
    return _normalise(weighted_sum, weight_sum, w.sum())


def selective_cross_entropy_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    softcap: float | None = None,
    chunk_rows: int = 4096,
    weight_sum: torch.Tensor | float | None = None,
    ignore_index: int = IGNORE_INDEX,
    checkpoint_chunks: bool = False,
) -> torch.Tensor:
    """Computes `Σ(w_i * CE_i) / weight_sum` over supervised positions without full `[B, T, V]` logits."""
    if hidden_states.size(1) <= 1:
        return hidden_states.sum() * 0.0

    h_shift = hidden_states[:, :-1, :]
    # With model_split (device_map="auto") the last decoder layer, and so `hidden_states`, sits on the last GPU
    # while labels/weights stay on the input device. Boolean indexing needs the mask on the indexed tensor's device.
    h_device = h_shift.device
    y_shift = labels[:, 1:].to(h_device)
    mask = y_shift.ne(ignore_index)

    n_sup = int(mask.sum().item())
    if n_sup == 0:
        return hidden_states.sum() * 0.0

    h_sel = h_shift[mask]  # [N, H]
    y_sel = y_shift[mask]  # [N]
    w_sel = weights[:, 1:].to(h_device)[mask].float() if weights is not None else None

    head_device = next(lm_head.parameters()).device
    step = max(1, int(chunk_rows))
    weighted_sum: torch.Tensor | None = None

    for start in range(0, n_sup, step):
        end = min(n_sup, start + step)
        h_chunk = h_sel[start:end]
        y_chunk = y_sel[start:end]
        if h_chunk.device != head_device:
            h_chunk = h_chunk.to(head_device)
        if y_chunk.device != head_device:
            y_chunk = y_chunk.to(head_device)
        w_chunk = w_sel[start:end] if w_sel is not None else None

        chunk_total = _run_chunk(lm_head, softcap, h_chunk, y_chunk, w_chunk, checkpoint_chunks)
        weighted_sum = chunk_total if weighted_sum is None else (weighted_sum + chunk_total)

    assert weighted_sum is not None
    fallback = w_sel.sum() if w_sel is not None else torch.tensor(float(n_sup), dtype=torch.float32)
    return _normalise(weighted_sum, weight_sum, fallback)


@dataclass
class WeightedLanguageModelingCollator:
    """Right-pads `input_ids`, `labels`, `attention_mask`, and optional per-token `weights`."""

    pad_token_id: int = 0
    ignore_index: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None  # bucket lengths so XLA compiles a bounded set of shapes

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("WeightedLanguageModelingCollator received an empty batch")
        max_len = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        bsz = len(examples)
        has_weights = "weights" in examples[0] and examples[0]["weights"] is not None

        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        weights = torch.zeros((bsz, max_len), dtype=torch.float32) if has_weights else None

        for i, ex in enumerate(examples):
            seq_ids = ex["input_ids"]
            seq_lbl = ex["labels"]
            n = len(seq_ids)
            input_ids[i, :n] = torch.as_tensor(seq_ids, dtype=torch.long)
            labels[i, :n] = torch.as_tensor(seq_lbl, dtype=torch.long)
            attention_mask[i, :n] = 1
            if weights is not None and ex.get("weights") is not None:
                weights[i, :n] = torch.as_tensor(ex["weights"], dtype=torch.float32)

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if weights is not None:
            batch["weights"] = weights
        return batch


def build_selective_sft_trainer_cls() -> type:
    """Lazily constructs `SelectiveLossSFTTrainer(SFTTrainer)` so importing `loss.py` stays lightweight."""
    from trl import SFTTrainer

    class SelectiveLossSFTTrainer(SFTTrainer):
        """`SFTTrainer` subclass using selective-logits softcapped CE with step-wide weight normalization."""

        def __init__(
            self,
            *args: Any,
            chunk_rows: int = 4096,
            loss_impl: str = "selective",
            checkpoint_chunks: bool = True,
            custom_train_sampler: Any | None = None,
            metrics_tracker: dict[str, float] | None = None,
            optimizer_factory: Any | None = None,
            tuple_outputs: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            # FSDP (incl. XLA FSDPv2) forbids passing pre-built optimizers: they must be created after the
            # model is wrapped, so the factory builds AdamW + WSD scheduler on the wrapped parameters.
            self._optimizer_factory = optimizer_factory
            # XLA FSDPv2's output-sharding hook only understands tensors/tuples/CausalLMOutputWithPast.
            self._tuple_outputs = bool(tuple_outputs)
            self.chunk_rows = int(chunk_rows)
            if loss_impl == "liger":
                logger.warning("loss.impl='liger' is not implemented; using the equivalent 'selective' loss")
                loss_impl = "selective"
            self.loss_impl = loss_impl
            self.checkpoint_chunks = bool(checkpoint_chunks)
            self.custom_train_sampler = custom_train_sampler
            self.metrics_tracker = metrics_tracker if metrics_tracker is not None else {}
            self.loss_is_scaled_for_ga = True
            self._softcap = get_model_softcap(self.model)

        def _set_signature_columns_if_needed(self) -> None:
            super()._set_signature_columns_if_needed()
            if self._signature_columns is not None and "weights" not in self._signature_columns:
                self._signature_columns = [*self._signature_columns, "weights"]

        def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
            if self._optimizer_factory is None:
                super().create_optimizer_and_scheduler(num_training_steps)
                return
            self.optimizer, self.lr_scheduler = self._optimizer_factory(self.model)

        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            if self.custom_train_sampler is not None:
                return self.custom_train_sampler
            return super()._get_train_sampler(train_dataset)

        def _get_num_items_in_batch(
            self, batch_samples: list[dict[str, Any]], device: torch.device
        ) -> torch.Tensor | None:
            if not batch_samples or "labels" not in batch_samples[0]:
                return None

            # Accumulate as tensors and read back once (one host sync per optimizer step, not per micro-batch).
            total_w = torch.zeros((), dtype=torch.float32, device=device)
            total_sup = torch.zeros((), dtype=torch.float32, device=device)
            total_all = torch.zeros((), dtype=torch.float32, device=device)
            for batch in batch_samples:
                lbl = batch["labels"]
                y_shift = lbl[..., 1:]
                mask = y_shift.ne(IGNORE_INDEX).to(device=device, dtype=torch.float32)
                total_sup = total_sup + mask.sum()
                if "attention_mask" in batch:
                    total_all = total_all + batch["attention_mask"].to(device=device, dtype=torch.float32).sum()
                else:
                    total_all = total_all + float(lbl.numel())

                if "weights" in batch and batch["weights"] is not None:
                    w_shift = batch["weights"][..., 1:].to(device=device, dtype=torch.float32)
                    total_w = total_w + (w_shift * mask).sum()
                else:
                    total_w = total_w + mask.sum()

            if self.args.world_size > 1 and self.args.average_tokens_across_devices:
                total_w = self.accelerator.gather(total_w).sum()

            stats = torch.stack([total_sup, total_all, total_w.detach().to(total_sup.device)]).cpu().tolist()
            self.metrics_tracker["step_supervised_tokens"] = float(stats[0])
            self.metrics_tracker["step_total_tokens"] = float(stats[1])
            self.metrics_tracker["step_weight_sum"] = float(stats[2])
            return total_w

        def compute_loss(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | int | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            weights = model_inputs.pop("weights", None)

            causal_lm = unwrap_causal_lm(model)
            backbone = getattr(causal_lm, "model", None)
            if backbone is None or not hasattr(causal_lm, "lm_head"):
                raise RuntimeError(f"Expected Gemma4 CausalLM with .model and .lm_head, got {type(causal_lm)}")

            captured: dict[str, torch.Tensor] = {}

            def _capture_hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                captured["hidden_states"] = h

            handle = backbone.register_forward_hook(_capture_hook)
            try:
                # Pass logits_to_keep=1 without labels so DDP/PEFT forward hooks execute cleanly
                # while `lm_head` inside `forward` only touches 1 token instead of all T tokens.
                if self._tuple_outputs:
                    model_inputs["return_dict"] = False
                outputs = model(**model_inputs, logits_to_keep=1)
            finally:
                handle.remove()

            hidden_states = captured["hidden_states"]
            # During evaluation (`not model.training`), `num_items_in_batch` is None -> per-batch mean
            self.loss_is_scaled_for_ga = num_items_in_batch is not None

            loss_fn = static_cross_entropy_loss if self.loss_impl == "static" else selective_cross_entropy_loss
            loss = loss_fn(
                hidden_states,
                causal_lm.lm_head,
                labels,
                weights=weights,
                softcap=self._softcap,
                chunk_rows=self.chunk_rows,
                weight_sum=num_items_in_batch,
                checkpoint_chunks=self.checkpoint_chunks,
            )

            if (
                model.training
                and self.args.world_size > 1
                and self.args.average_tokens_across_devices
                and num_items_in_batch is not None
            ):
                loss = loss * self.accelerator.num_processes

            return (loss, outputs) if return_outputs else loss

    return SelectiveLossSFTTrainer
