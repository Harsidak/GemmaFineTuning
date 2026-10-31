"""Training callbacks: WSD stop, wall-clock checkpointing, val interval, dead-adapter check, metrics.

Implements design_doc.md §7.8 and §7.9:
- Appends structured JSONL metrics to `runs/<run_id>/metrics/<adapter>.jsonl` every `log_every_steps`.
- Emits throughput projection at `throughput_project_step` (default step 10).
- Runs dead-adapter check (`‖lora_B‖ == 0`) at `dead_check_step` (default step 50) without aborting.
- Triggers validation every `val_every_min` wall-clock minutes.
- Saves resumable wall-clock checkpoints every `checkpoint_every_min` minutes under
  `runs/<run_id>/checkpoints/<adapter>/step_<n>/`, retaining the last `keep_checkpoints`.
  Step numbers are *global* (resumed offset + segment step), so checkpoints written after a resume sort
  after the ones they resumed from.
- Aborts with `NonFiniteLossError` after `max_nonfinite_logs` consecutive NaN/Inf loss or grad-norm logs, so the
  orchestrator restarts from the last good checkpoint instead of training on (and saving) corrupt weights.
- Stops cleanly when `stop_requested()` turns true (SIGTERM from the orchestrator's deadline).
- Multi-process (FSDP/DDP): every time-based or signal-based decision (stop, validate, checkpoint) is taken from rank
  0's clock and an any-rank signal flag, so all ranks run the same collectives; under FSDP the LoRA weights and
  optimizer state are gathered on every rank before rank 0 writes them (see `distributed.py`).
"""

from __future__ import annotations

import json
import logging
import re
import shutil  # noqa: F401 - tests patch `callbacks.shutil.disk_usage`
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..shared.checkpoints import (  # noqa: F401 - re-exported: the orchestrator, export phase and tests import them from here
    checkpoint_is_loadable,
    find_latest_checkpoint,
    list_checkpoints,
    write_checkpoint,
)
from ..shared.guards import NonFiniteGuard, NonFiniteLossError  # noqa: F401
from ..shared.sysmon import read_latest, wandb_system_metrics
from .device import capture_rng_state, cpu_state_dict, peak_memory_mb, save_object
from .distributed import (
    SyncedClock,
    agree_any,
    fsdp_full_optim_state_dict,
    gather_lora_state_dict,
)
from .loss import unwrap_causal_lm
from .schedule import WSDScheduleController

logger = logging.getLogger(__name__)


def save_adapter_weights(
    model: Any, out_dir: Path, accelerator: str = "cpu", state_dict: dict[str, Any] | None = None
) -> None:
    """PEFT `save_pretrained`; on XLA the LoRA tensors are synced and copied to host first.

    The host copy keeps the raw (adapter-name-qualified) keys so PEFT's own filtering in `save_pretrained`
    still selects them; passing an already-converted PEFT state dict would filter everything out. `state_dict` is the
    LoRA tensors already gathered from an FSDP-sharded model (`distributed.gather_lora_state_dict`).
    """
    unwrapped = model.module if hasattr(model, "module") else model
    if state_dict is not None:
        unwrapped.save_pretrained(out_dir, state_dict=state_dict)
        return
    if accelerator != "tpu":
        unwrapped.save_pretrained(out_dir)
        return
    raw = {k: v for k, v in unwrapped.state_dict().items() if "lora_" in k}
    unwrapped.save_pretrained(out_dir, state_dict=cpu_state_dict(raw, accelerator))


def inspect_dead_lora_modules(model: Any, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """Checks `‖lora_B‖` across all LoRA modules, grouping any exact-zero B weights (§7.9).

    `state_dict` (gathered LoRA tensors) replaces the model's own parameters, which are flat shards under FSDP."""
    cur = model.module if hasattr(model, "module") else model
    zero_by_proj: dict[str, list[int]] = defaultdict(list)
    total_modules = 0
    zero_count = 0

    params = state_dict.items() if state_dict is not None else cur.named_parameters()
    for name, param in params:
        if ".lora_B." not in name:
            continue
        total_modules += 1
        norm_val = float(param.detach().float().norm().cpu().item())
        if norm_val == 0.0:
            zero_count += 1
            m_layer = re.search(r"\.layers\.(\d+)\.(.+?)\.lora_B\.", name)
            if m_layer:
                zero_by_proj[m_layer.group(2)].append(int(m_layer.group(1)))
            else:
                zero_by_proj[name].append(-1)

    grouped: dict[str, str] = {}
    for proj, layers in sorted(zero_by_proj.items()):
        valid_layers = sorted(x for x in layers if x >= 0)
        if not valid_layers:
            grouped[proj] = "non-indexed"
        elif len(valid_layers) == 1:
            grouped[proj] = f"layer {valid_layers[0]}"
        else:
            grouped[proj] = f"layers {valid_layers[0]}..{valid_layers[-1]} ({len(valid_layers)} layers)"

    return {
        "total_lora_b_modules": total_modules,
        "dead_lora_b_modules": zero_count,
        "dead_grouped": grouped,
    }


def save_wallclock_checkpoint(
    *,
    checkpoint_root: Path,
    step: int,
    epoch: float,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    wsd_controller: WSDScheduleController,
    sampler: Any | None,
    samples_in_epoch: int,
    keep_checkpoints: int,
    is_main_process: bool = True,
    accelerator: str = "cpu",
    min_free_disk_gib: float = 0.0,
    reason: str = "wallclock",
    extra_meta: dict[str, Any] | None = None,
    progress_fn: Callable[[Path], None] | None = None,
    adapter_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path | None:
    """Writes a self-contained resumable checkpoint directory `step_<n>/` and prunes old ones (§7.8).

    Publishing (temp dir, post-write verification, atomic rename, pruning, disk-space guard) is the shared
    `write_checkpoint`; this function only supplies the PyTorch-specific state. `extra_meta` is merged into
    `trainer_resume.json` (fingerprints, best-loss state) and `progress_fn(tmp_dir)` writes the task ledger.
    Returns None when nothing was written (not the main process, or the disk is too full).
    `adapter_state` / `optimizer_state` are pre-gathered from an FSDP model; the optimizer state is then the
    name-keyed full state (`optimizer_format: fsdp_full` in `trainer_resume.json`) rather than `optimizer.state_dict()`.
    """
    if not is_main_process:
        return None

    def _write(tmp_dir: Path) -> None:
        save_adapter_weights(model, tmp_dir / "adapter", accelerator, state_dict=adapter_state)
        if optimizer_state is not None:
            torch.save(optimizer_state, tmp_dir / "optimizer.pt")
        elif optimizer is not None:
            save_object(optimizer.state_dict(), tmp_dir / "optimizer.pt", accelerator)
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        torch.save(capture_rng_state(accelerator), tmp_dir / "rng_state.pt")
        if progress_fn is not None:
            progress_fn(tmp_dir)
        meta = {
            "step": int(step),
            "epoch": float(epoch),
            "elapsed_sec": wsd_controller.total_elapsed_sec(),
            "schedule_state": wsd_controller.state_dict(),
            "sampler_state": sampler.state_dict(samples_in_epoch) if sampler is not None else None,
            "saved_at_ts": time.time(),
            "reason": reason,
            "accelerator": accelerator,
            "optimizer_format": "fsdp_full" if optimizer_state is not None else "torch",
            **(extra_meta or {}),
        }
        (tmp_dir / "trainer_resume.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if adapter_state is not None:  # FSDP parameters are shards; size the estimate from the gathered tensors
        trainable_bytes = 3 * sum(t.numel() * t.element_size() for t in adapter_state.values())
    else:
        trainable_bytes = 4 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return write_checkpoint(
        checkpoint_root,
        step,
        _write,
        keep=keep_checkpoints,
        min_free_disk_gib=min_free_disk_gib,
        fallback_expected_bytes=trainable_bytes,
        reason=reason,
    )


class MultiLoraTrainingCallback(TrainerCallback):
    """Coordinates WSD timing, metrics JSONL, throughput projection, dead-LoRA check, and checkpoints."""

    def __init__(
        self,
        *,
        adapter_name: str,
        wsd_controller: WSDScheduleController,
        metrics_jsonl_path: Path,
        checkpoint_root: Path,
        metrics_tracker: dict[str, float],
        sampler: Any | None,
        dataset_total_tokens: int,
        samples_per_step: int,
        log_every_steps: int = 5,
        val_every_min: float = 60.0,
        checkpoint_every_min: float = 45.0,
        keep_checkpoints: int = 2,
        dead_check_step: int = 50,
        throughput_project_step: int = 10,
        has_eval_dataset: bool = False,
        step_offset: int = 0,
        accelerator: str = "cpu",
        min_free_disk_gib: float = 0.0,
        max_nonfinite_logs: int = 3,
        stop_requested: Callable[[], bool] | None = None,
        session: Any | None = None,
        clock: SyncedClock | None = None,
    ) -> None:
        self.session = session
        # Rank-0 wall clock shared by all ranks (also the WSD controller's now_fn); None = local time.
        self.clock = clock
        self._gathered_lora: dict[str, Any] | None = None
        self.adapter_name = adapter_name
        self.step_offset = max(0, int(step_offset))
        self.accelerator = accelerator
        self.min_free_disk_gib = float(min_free_disk_gib)
        self.max_nonfinite_logs = max(1, int(max_nonfinite_logs))
        self.stop_requested = stop_requested
        # One guard implementation for every backend: the session's when there is one, else a private instance.
        self._guard = session.nonfinite if session is not None else NonFiniteGuard(self.max_nonfinite_logs, f"[{adapter_name}]")
        self.stopped_by_signal = False
        self.wsd = wsd_controller
        self.metrics_jsonl_path = metrics_jsonl_path
        self.checkpoint_root = checkpoint_root
        self.metrics_tracker = metrics_tracker
        self.sampler = sampler
        self.dataset_total_tokens = max(1, int(dataset_total_tokens))
        self.samples_per_step = max(1, int(samples_per_step))
        self.log_every_steps = max(1, int(log_every_steps))
        self.val_every_sec = max(1.0, float(val_every_min) * 60.0)
        self.ckpt_every_sec = max(1.0, float(checkpoint_every_min) * 60.0)
        self.keep_checkpoints = max(1, int(keep_checkpoints))
        self.dead_check_step = max(1, int(dead_check_step))
        self.throughput_project_step = max(1, int(throughput_project_step))
        self.has_eval_dataset = has_eval_dataset

        self._last_ckpt_ts = self.wsd.now()
        self._last_val_ts = self.wsd.now()
        self._window_start_ts = self.wsd.now()
        self._window_all_tokens = 0.0
        self._window_sup_tokens = 0.0
        self.cumulative_all_tokens = 0
        self.cumulative_sup_tokens = 0
        self.samples_in_epoch = 0
        # A resumed sampler starts mid-epoch; count from that offset so the next checkpoint's offset is absolute.
        self._pending_resume_offset = int(getattr(sampler, "resume_offset", 0) or 0)

        self.dead_check_result: dict[str, Any] | None = None
        self.throughput_projection: dict[str, Any] | None = None
        self.val_losses: list[dict[str, float]] = []
        self.train_losses: list[dict[str, float]] = []
        self._trainer_ref: Any = None

    @property
    def nonfinite_streak(self) -> int:
        return self._guard.streak

    def bind_trainer(self, trainer: Any) -> None:
        self._trainer_ref = trainer

    def _save_best(self, tmp_dir: Path) -> None:
        save_adapter_weights(self._trainer_ref.model, tmp_dir, self.accelerator, state_dict=self._gathered_lora)

    def _progress_writer(self, state: TrainerState) -> Callable[[Path], None] | None:
        """Writes the task ledger (which examples this epoch has consumed) next to a checkpoint."""
        sess, smp = self.session, self.sampler
        if sess is None or smp is None or not hasattr(smp, "consumed_indices"):
            return None

        def _write(tmp_dir: Path) -> None:
            consumed = [sess.keys[i] for i in smp.consumed_indices(self.samples_in_epoch) if i < len(sess.keys)]
            sess.write_progress(tmp_dir, epoch=smp._order_epoch, consumed_keys=consumed)

        return _write

    def global_step(self, state: TrainerState) -> int:
        return self.step_offset + int(state.global_step)

    def save_checkpoint(self, state: TrainerState, model: Any, *, reason: str) -> Path | None:
        """Writes a resumable checkpoint at the current global step (used by timers, SIGTERM, and crashes)."""
        if self._trainer_ref is None or model is None:
            return None
        # Collective under FSDP (every rank reaches this from the same agreed decision); None for other models.
        adapter_state = gather_lora_state_dict(model)
        optimizer_state = (
            fsdp_full_optim_state_dict(model, self._trainer_ref.optimizer) if adapter_state is not None else None
        )
        return save_wallclock_checkpoint(
            checkpoint_root=self.checkpoint_root,
            step=self.global_step(state),
            epoch=float(state.epoch or 0.0),
            model=model,
            optimizer=self._trainer_ref.optimizer,
            scheduler=self._trainer_ref.lr_scheduler,
            wsd_controller=self.wsd,
            sampler=self.sampler,
            samples_in_epoch=self.samples_in_epoch,
            keep_checkpoints=self.keep_checkpoints,
            is_main_process=state.is_world_process_zero,
            accelerator=self.accelerator,
            min_free_disk_gib=self.min_free_disk_gib,
            reason=reason,
            extra_meta=self.session.checkpoint_meta() if self.session is not None else None,
            progress_fn=self._progress_writer(state),
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )

    def _dead_check(self, model: Any) -> dict[str, Any]:
        gathered = gather_lora_state_dict(model)  # collective under FSDP
        if gathered is not None and not gathered:  # non-zero rank: rank 0 holds the tensors and reports
            return {"total_lora_b_modules": 0, "dead_lora_b_modules": 0, "dead_grouped": {}, "reported_by": "rank 0"}
        return inspect_dead_lora_modules(model, state_dict=gathered)

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        self.samples_in_epoch = self._pending_resume_offset
        self._pending_resume_offset = 0
        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        # Same timestamp on every rank (one tiny broadcast per optimizer step in multi-process runs).
        now = self.clock.sync() if self.clock is not None else self.wsd.now()
        self.wsd.on_step_end(self.global_step(state), now=now)

        step_all = float(self.metrics_tracker.get("step_total_tokens", 0.0)) * max(1, args.world_size)
        step_sup = float(self.metrics_tracker.get("step_supervised_tokens", 0.0)) * max(1, args.world_size)
        self._window_all_tokens += step_all
        self._window_sup_tokens += step_sup
        self.cumulative_all_tokens += int(step_all)
        self.cumulative_sup_tokens += int(step_sup)
        self.samples_in_epoch += self.samples_per_step

        model = kwargs.get("model")
        is_main = state.is_world_process_zero

        # Dead-adapter check (§7.9)
        if state.global_step == self.dead_check_step and model is not None and self.dead_check_result is None:
            self.dead_check_result = self._dead_check(model)
            if is_main and self.dead_check_result["dead_lora_b_modules"] > 0:
                logger.warning(
                    "[%s] Dead-adapter check at step %d: %d/%d lora_B modules still zero: %s",
                    self.adapter_name,
                    state.global_step,
                    self.dead_check_result["dead_lora_b_modules"],
                    self.dead_check_result["total_lora_b_modules"],
                    self.dead_check_result["dead_grouped"],
                )

        # Throughput projection (§7.9)
        if state.global_step == self.throughput_project_step and self.throughput_projection is None:
            seg_elapsed = max(1e-6, now - self.wsd.t_start)
            toks_per_sec = self.cumulative_all_tokens / seg_elapsed
            budget_sec = max(0.0, self.wsd.deadline - self.wsd.t_start)
            projected_passes = (toks_per_sec * budget_sec) / float(self.dataset_total_tokens)
            proj_t_end = self.wsd.projected_t_end(now)
            self.throughput_projection = {
                "measured_at_step": state.global_step,
                "tokens_per_sec": round(toks_per_sec, 2),
                "projected_dataset_passes_in_budget": round(projected_passes, 2),
                "projected_end_offset_sec": round(proj_t_end - self.wsd.t_start, 1),
            }
            if is_main:
                logger.info(
                    "[%s] Throughput @ step %d: %.1f tok/s | budget allows ~%.2f data passes | est remaining %.1f min",
                    self.adapter_name,
                    state.global_step,
                    toks_per_sec,
                    projected_passes,
                    max(0.0, proj_t_end - now) / 60.0,
                )

        # Wall-clock validation trigger
        if self.has_eval_dataset and (now - self._last_val_ts) >= self.val_every_sec:
            control.should_evaluate = True
            self._last_val_ts = now

        # Wall-clock checkpoint trigger (§7.8)
        if (now - self._last_ckpt_ts) >= self.ckpt_every_sec and self._trainer_ref is not None and model is not None:
            self.save_checkpoint(state, model, reason="wallclock")
            self._last_ckpt_ts = now

        # External stop (SIGTERM at the orchestrator's hard deadline): stop now, the caller saves final weights.
        # Signals reach ranks at slightly different times; any rank having one stops all of them at this step.
        signalled = self.stop_requested() if self.stop_requested is not None else False
        if self.clock is not None and self.clock.distributed:
            signalled = agree_any(signalled)
        if signalled:
            if is_main and not self.stopped_by_signal:
                logger.warning("[%s] Stop requested by signal at step %d", self.adapter_name, self.global_step(state))
            self.stopped_by_signal = True
            control.should_training_stop = True
            return control

        # Time-budget / WSD stop check (§7.6)
        if self.wsd.should_stop(now):
            if is_main:
                logger.info(
                    "[%s] Stopping cleanly at step %d (elapsed=%.1fs, decay_started=%s)",
                    self.adapter_name,
                    state.global_step,
                    self.wsd.total_elapsed_sec(now),
                    self.wsd.decay_started,
                )
            control.should_training_stop = True

        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        # NaN/Inf + divergence guards run on every rank so all ranks abort together (requires logging_nan_inf_filter=False).
        if "loss" in logs:
            gn = logs.get("grad_norm")
            gn_val = float(gn) if gn is not None else None
            if self.session is not None:
                self.session.check_health(self.global_step(state), float(logs["loss"]), gn_val)
            else:
                self._guard.observe(float(logs["loss"]), gn_val, self.global_step(state))

        # Every rank sees the same (already reduced) eval loss, so every rank reaches the same best/early-stop
        # decision; under FSDP the LoRA weights are gathered first because saving the best adapter is collective.
        if "eval_loss" in logs and self.session is not None:
            model = self._trainer_ref.model if self._trainer_ref is not None else None
            self._gathered_lora = gather_lora_state_dict(model)
            try:
                decision = self.session.on_eval(self.global_step(state), float(logs["eval_loss"]), self._save_best)
            finally:
                self._gathered_lora = None
            if decision.should_stop:
                control.should_training_stop = True

        if not state.is_world_process_zero:
            return

        now = self.wsd.now()
        if "eval_loss" in logs:
            self.val_losses.append(
                {
                    "step": state.global_step,
                    "elapsed_sec": round(self.wsd.total_elapsed_sec(now), 2),
                    "eval_loss": float(logs["eval_loss"]),
                }
            )

        if "loss" not in logs:
            return

        dt = max(1e-6, now - self._window_start_ts)
        tok_s = self._window_all_tokens / dt
        sup_tok_s = self._window_sup_tokens / dt
        self._window_start_ts = now
        self._window_all_tokens = 0.0
        self._window_sup_tokens = 0.0

        gpu_peak_mb = peak_memory_mb(self.accelerator)

        entry = {
            "step": self.global_step(state),
            "segment_step": state.global_step,
            "epoch": round(float(state.epoch or 0.0), 4),
            "elapsed_sec": round(self.wsd.total_elapsed_sec(now), 2),
            "lr": float(logs.get("learning_rate", 0.0)),
            "train_loss": float(logs["loss"]),
            "grad_norm": float(logs["grad_norm"]) if logs.get("grad_norm") is not None else None,
            "tokens_per_sec": round(tok_s, 2),
            "supervised_tokens_per_sec": round(sup_tok_s, 2),
            "supervised_tokens_in_step": int(self.metrics_tracker.get("step_supervised_tokens", 0.0)),
            "peak_gpu_mem_mb": gpu_peak_mb,
        }
        self.train_losses.append({"step": state.global_step, "train_loss": entry["train_loss"]})
        if self.session is not None:
            self.session.log_metrics(
                entry["step"],
                {"train/loss": entry["train_loss"], "train/grad_norm": entry["grad_norm"], "train/lr": entry["lr"],
                 "train/epoch": entry["epoch"], "perf/tokens_per_sec": entry["tokens_per_sec"],
                 "perf/supervised_tokens_per_sec": entry["supervised_tokens_per_sec"], "perf/peak_mem_mb": entry["peak_gpu_mem_mb"],
                 **wandb_system_metrics(read_latest(self.metrics_jsonl_path.parent / "system.jsonl"))},
            )
        self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def finalize_checks(self, model: Any) -> None:
        """Ensures dead-adapter check still runs at training end if fewer than `dead_check_step` steps ran."""
        if self.dead_check_result is None and model is not None:
            _ = unwrap_causal_lm(model)
            self.dead_check_result = self._dead_check(model)
