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
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
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

from .device import capture_rng_state, cpu_state_dict, peak_memory_mb, save_object
from .loss import unwrap_causal_lm
from .schedule import WSDScheduleController

logger = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    """Training produced NaN/Inf loss or gradients for several consecutive logging windows."""


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def save_adapter_weights(model: Any, out_dir: Path, accelerator: str = "cpu") -> None:
    """PEFT `save_pretrained`; on XLA the LoRA tensors are synced and copied to host first.

    The host copy keeps the raw (adapter-name-qualified) keys so PEFT's own filtering in `save_pretrained`
    still selects them; passing an already-converted PEFT state dict would filter everything out.
    """
    unwrapped = model.module if hasattr(model, "module") else model
    if accelerator != "tpu":
        unwrapped.save_pretrained(out_dir)
        return
    raw = {k: v for k, v in unwrapped.state_dict().items() if "lora_" in k}
    unwrapped.save_pretrained(out_dir, state_dict=cpu_state_dict(raw, accelerator))


def inspect_dead_lora_modules(model: Any) -> dict[str, Any]:
    """Checks `‖lora_B‖` across all LoRA modules, grouping any exact-zero B weights (§7.9)."""
    cur = model.module if hasattr(model, "module") else model
    zero_by_proj: dict[str, list[int]] = defaultdict(list)
    total_modules = 0
    zero_count = 0

    for name, param in cur.named_parameters():
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
) -> Path | None:
    """Writes a self-contained resumable checkpoint directory `step_<n>/` and prunes old ones (§7.8).

    Disk safety (Kaggle `/kaggle/working` is ~20 GiB): before writing, the expected size (the previous
    checkpoint's size, else 4x the adapter) is compared with free space. Older checkpoints beyond the newest
    one are pruned first if needed; if space is still short the save is skipped with a warning instead of
    crashing training with `ENOSPC`. Returns None when nothing was written.
    """
    if not is_main_process:
        return None

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing = list_checkpoints(checkpoint_root)
    expected = _dir_size_bytes(existing[-1]) if existing else 0
    if not expected:
        expected = 4 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    floor = int(min_free_disk_gib * 1024**3)
    if shutil.disk_usage(checkpoint_root).free - expected < floor:
        for old in existing[:-1]:
            shutil.rmtree(old, ignore_errors=True)
        if shutil.disk_usage(checkpoint_root).free - expected < floor:
            logger.warning(
                "Skipping %s checkpoint at step %d: %.2f GiB free, need ~%.2f GiB + %.1f GiB floor",
                reason,
                step,
                shutil.disk_usage(checkpoint_root).free / 1024**3,
                expected / 1024**3,
                min_free_disk_gib,
            )
            return None

    ckpt_dir = checkpoint_root / f"step_{step:06d}"
    tmp_dir = checkpoint_root / f".tmp_step_{step:06d}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_adapter_weights(model, tmp_dir / "adapter", accelerator)
        if optimizer is not None:
            save_object(optimizer.state_dict(), tmp_dir / "optimizer.pt", accelerator)
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        torch.save(capture_rng_state(accelerator), tmp_dir / "rng_state.pt")

        meta = {
            "step": int(step),
            "epoch": float(epoch),
            "elapsed_sec": wsd_controller.total_elapsed_sec(),
            "schedule_state": wsd_controller.state_dict(),
            "sampler_state": sampler.state_dict(samples_in_epoch) if sampler is not None else None,
            "saved_at_ts": time.time(),
            "reason": reason,
            "accelerator": accelerator,
        }
        (tmp_dir / "trainer_resume.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Atomic publish: the step dir only ever exists in complete form.
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp_dir.rename(ckpt_dir)

    existing = list_checkpoints(checkpoint_root)
    while len(existing) > max(1, int(keep_checkpoints)):
        shutil.rmtree(existing.pop(0), ignore_errors=True)

    return ckpt_dir


def list_checkpoints(checkpoint_root: Path) -> list[Path]:
    """Complete `step_*` checkpoints sorted oldest -> newest by global step."""
    if not checkpoint_root.exists():
        return []
    candidates = [
        d
        for d in checkpoint_root.glob("step_*")
        if d.is_dir()
        and (d / "trainer_resume.json").is_file()
        and (d / "adapter" / "adapter_model.safetensors").is_file()
    ]
    return sorted(candidates, key=lambda d: int(d.name.split("_", 1)[1]))


def find_latest_checkpoint(checkpoint_root: Path) -> Path | None:
    found = list_checkpoints(checkpoint_root)
    return found[-1] if found else None


def checkpoint_is_loadable(ckpt: Path) -> bool:
    """Cheap integrity check: resume metadata parses and the safetensors header is readable."""
    try:
        json.loads((ckpt / "trainer_resume.json").read_text(encoding="utf-8"))
        from safetensors import safe_open

        with safe_open(str(ckpt / "adapter" / "adapter_model.safetensors"), framework="pt") as fh:
            return len(list(fh.keys())) > 0
    except Exception as exc:  # noqa: BLE001 - any failure means "do not resume from this one"
        logger.warning("Checkpoint %s is unreadable (%s); skipping it", ckpt, exc)
        return False


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
    ) -> None:
        self.adapter_name = adapter_name
        self.step_offset = max(0, int(step_offset))
        self.accelerator = accelerator
        self.min_free_disk_gib = float(min_free_disk_gib)
        self.max_nonfinite_logs = max(1, int(max_nonfinite_logs))
        self.stop_requested = stop_requested
        self.nonfinite_streak = 0
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

    def bind_trainer(self, trainer: Any) -> None:
        self._trainer_ref = trainer

    def global_step(self, state: TrainerState) -> int:
        return self.step_offset + int(state.global_step)

    def save_checkpoint(self, state: TrainerState, model: Any, *, reason: str) -> Path | None:
        """Writes a resumable checkpoint at the current global step (used by timers, SIGTERM, and crashes)."""
        if self._trainer_ref is None or model is None:
            return None
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
        )

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        self.samples_in_epoch = self._pending_resume_offset
        self._pending_resume_offset = 0
        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        now = self.wsd.now()
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
            self.dead_check_result = inspect_dead_lora_modules(model)
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
        if self.stop_requested is not None and self.stop_requested():
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

        # NaN/Inf guard runs on every rank so all ranks abort together (requires logging_nan_inf_filter=False).
        if "loss" in logs:
            loss_val = float(logs["loss"])
            gn = logs.get("grad_norm")
            bad = not math.isfinite(loss_val) or (gn is not None and not math.isfinite(float(gn)))
            self.nonfinite_streak = self.nonfinite_streak + 1 if bad else 0
            if bad:
                logger.warning(
                    "[%s] Non-finite loss/grad_norm at step %d (loss=%s, grad_norm=%s) [%d/%d]",
                    self.adapter_name,
                    self.global_step(state),
                    loss_val,
                    gn,
                    self.nonfinite_streak,
                    self.max_nonfinite_logs,
                )
            if self.nonfinite_streak >= self.max_nonfinite_logs:
                raise NonFiniteLossError(
                    f"[{self.adapter_name}] {self.nonfinite_streak} consecutive non-finite loss/grad-norm logs at "
                    f"step {self.global_step(state)}; aborting so the run restarts from the last good checkpoint. "
                    "Lower train.lr, raise schedule.warmup_steps, or check data for empty supervision."
                )

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
        self.metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def finalize_checks(self, model: Any) -> None:
        """Ensures dead-adapter check still runs at training end if fewer than `dead_check_step` steps ran."""
        if self.dead_check_result is None and model is not None:
            _ = unwrap_causal_lm(model)
            self.dead_check_result = inspect_dead_lora_modules(model)
