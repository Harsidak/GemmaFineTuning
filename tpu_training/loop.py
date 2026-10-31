"""The training loop: schedule, sampling, guards, wall-clock checkpoints, validation, stop handling.

Everything that is not JAX lives on the shared side (`TrainingSession`, `write_checkpoint`, `WSDScheduleController`,
`LengthGroupedLongestFirstSampler`), so a step here and a step of the PyTorch backend are governed by the same code:
same time-aware WSD learning rate, same NaN / divergence guards, same checkpoint layout, same task ledger.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.multi_lora_sft.shared.checkpoints import write_checkpoint
from pipeline.multi_lora_sft.shared.sysmon import read_latest, wandb_system_metrics

from .batching import batch_stats, collate
from .peft_io import OPT_FILE, write_adapter_dir

logger = logging.getLogger(__name__)


def _peak_device_mem_mb() -> float | None:
    """Peak bytes in use on the first local accelerator (TPU HBM), from inside the process that owns it."""
    try:
        import jax

        stats = jax.local_devices()[0].memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        return round(peak / 1024**2, 1) if peak else None
    except Exception:  # noqa: BLE001 - CPU backends have no memory_stats
        return None


@dataclass
class LoopSettings:
    batch_size: int
    grad_accum: int
    base_lr: float
    pad_id: int
    pad_multiple: int | None
    remaining_steps: int  # optimizer steps this segment may run
    step_offset: int  # steps already done in earlier sessions (global step = offset + segment step)
    r: int
    alpha: float
    use_rslora: bool
    base_model_id: str = ""
    target_regex: str | None = None
    log_every: int = 5
    val_every_sec: float = 3600.0
    ckpt_every_sec: float = 2700.0
    keep_checkpoints: int = 2
    min_free_disk_gib: float = 2.0
    dead_check_step: int = 50
    throughput_project_step: int = 10
    emergency_checkpoint: bool = True


@dataclass
class LoopResult:
    segment_steps: int = 0
    final_train_loss: float | None = None
    epoch: float = 0.0
    stopped_by_signal: bool = False
    stopped_early: bool = False
    skipped_nonfinite_steps: int = 0
    dead_check_result: dict[str, Any] | None = None
    throughput_projection: dict[str, Any] | None = None
    val_losses: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    supervised_tokens: int = 0


class TrainLoop:
    def __init__(
        self,
        trainer: Any,
        session: Any,
        wsd: Any,
        sampler: Any,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        settings: LoopSettings,
        *,
        ckpt_root: Path,
        metrics_path: Path,
        stop_requested: Callable[[], bool] | None = None,
        dataset_total_tokens: int = 0,
    ) -> None:
        self.t, self.session, self.wsd, self.sampler = trainer, session, wsd, sampler
        self.rows, self.val_rows, self.s = train_rows, val_rows, settings
        self.ckpt_root, self.metrics_path = Path(ckpt_root), Path(metrics_path)
        self.stop_requested = stop_requested
        self.dataset_total_tokens = dataset_total_tokens
        self.res = LoopResult()
        self._buf: list[int] = []
        self._epoch = int(getattr(sampler, "epoch", 0))
        self._samples_in_epoch = 0
        self._val_batches: list[dict[str, Any]] | None = None
        self._global_step = settings.step_offset
        self._loss_sum, self._loss_n = 0.0, 0
        self._win_loss, self._win_n, self._win_tokens, self._win_t0 = 0.0, 0, 0, time.time()

    # ------------------------------------------------------------------ data
    def _take(self, n: int) -> list[int]:
        """Next up to `n` indices of the current epoch; opens the next epoch when the previous one is used up."""
        for _ in range(3):
            if self._buf:
                break
            self._buf = list(iter(self.sampler))  # the sampler applies the ledger exclusion for the resumed epoch
            self._epoch = int(getattr(self.sampler, "_order_epoch", self._epoch + 1))
            self._samples_in_epoch = 0
        taken, self._buf = self._buf[:n], self._buf[n:]
        return taken

    def _micro_batches(self, idxs: list[int]) -> list[dict[str, Any]]:
        bs = self.s.batch_size
        chunks = [idxs[i : i + bs] for i in range(0, len(idxs), bs)]
        return [
            collate([self.rows[i] for i in c], batch_size=bs, pad_id=self.s.pad_id, pad_multiple=self.s.pad_multiple)
            for c in chunks
        ]

    # ------------------------------------------------------------------ validation / checkpoints
    def evaluate(self) -> float | None:
        if not self.val_rows:
            return None
        if self._val_batches is None:
            order = sorted(range(len(self.val_rows)), key=lambda i: len(self.val_rows[i]["input_ids"]))
            bs = self.s.batch_size
            self._val_batches = [
                collate([self.val_rows[i] for i in order[j : j + bs]], batch_size=bs, pad_id=self.s.pad_id, pad_multiple=self.s.pad_multiple)
                for j in range(0, len(order), bs)
            ]
        return self.t.eval_loss(self._val_batches)

    def _write_adapter(self, out_dir: Path, arrays: dict[str, Any] | None = None) -> None:
        write_adapter_dir(
            out_dir, arrays if arrays is not None else self.t.lora_numpy(), r=self.s.r, alpha=self.s.alpha,
            use_rslora=self.s.use_rslora, base_model=self.s.base_model_id, target_regex=self.s.target_regex,
        )

    def save_checkpoint(self, reason: str) -> Path | None:
        arrays, opt = self.t.lora_numpy(), self.t.opt_leaves()  # host copies first: the writer may be slow, training state moves on
        consumed = []
        if hasattr(self.sampler, "consumed_indices"):
            consumed = [self.session.keys[i] for i in self.sampler.consumed_indices(self._samples_in_epoch) if i < len(self.session.keys)]
        step, n_rows = self._global_step, max(1, len(self.rows))
        extra = self.session.checkpoint_meta()

        def _write(tmp: Path) -> None:
            self._write_adapter(tmp / "adapter", arrays)
            from .peft_io import save_opt_state

            save_opt_state(tmp / OPT_FILE, opt)
            self.session.write_progress(tmp, epoch=self._epoch, consumed_keys=consumed)
            meta = {
                "step": step,
                "epoch": float(self._epoch) + self._samples_in_epoch / n_rows,
                "elapsed_sec": self.wsd.total_elapsed_sec(),
                "schedule_state": self.wsd.state_dict(),
                "sampler_state": self.sampler.state_dict(self._samples_in_epoch),
                "saved_at_ts": time.time(),
                "reason": reason,
                "accelerator": self.session.accelerator,
                **extra,
            }
            (tmp / "trainer_resume.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return write_checkpoint(
            self.ckpt_root, step, _write, keep=self.s.keep_checkpoints, min_free_disk_gib=self.s.min_free_disk_gib,
            fallback_expected_bytes=int(self.t.n_params * 4 * 3), reason=reason,
        )

    def _validate(self, now: float) -> bool:
        """Runs validation; returns True when early stopping asks to end training."""
        loss = self.evaluate()
        if loss is None:
            return False
        decision = self.session.on_eval(self._global_step, float(loss), lambda tmp: self._write_adapter(Path(tmp)))
        self.res.val_losses.append(
            {"step": self._global_step, "elapsed_sec": round(self.wsd.total_elapsed_sec(now), 2), "eval_loss": float(loss)}
        )
        logger.info("[%s] val_loss=%.5f @ step %d", self.session.adapter, loss, self._global_step)
        return bool(decision.should_stop)

    # ------------------------------------------------------------------ logging
    def _log(self, step_res: dict[str, Any], lr: float, now: float) -> None:
        n = max(1, self._win_n)
        loss = self._win_loss / n
        dt = max(1e-6, now - self._win_t0)
        entry = {
            "step": self._global_step,
            "segment_step": self.res.segment_steps,
            "epoch": round(float(self._epoch) + self._samples_in_epoch / max(1, len(self.rows)), 4),
            "elapsed_sec": round(self.wsd.total_elapsed_sec(now), 2),
            "lr": lr,
            "train_loss": loss,
            "grad_norm": step_res["grad_norm"],
            "tokens_per_sec": round(self._win_tokens / dt, 2),
            "supervised_tokens_in_step": int(step_res.get("supervised", 0)),
            "peak_gpu_mem_mb": _peak_device_mem_mb(),  # TPU HBM on device 0 (same column the GPU backend fills)
            "skipped_nonfinite_steps": self.res.skipped_nonfinite_steps,
        }
        self.session.spike.observe(loss, self._global_step)
        self.session.log_metrics(
            self._global_step,
            {"train/loss": loss, "train/grad_norm": entry["grad_norm"], "train/lr": lr, "train/epoch": entry["epoch"],
             "train/skipped_nonfinite_steps": self.res.skipped_nonfinite_steps,
             "perf/tokens_per_sec": entry["tokens_per_sec"], "perf/peak_mem_mb": entry["peak_gpu_mem_mb"],
             **wandb_system_metrics(read_latest(self.metrics_path.parent / "system.jsonl"))},
        )
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:  # metrics are best-effort; never fail a step over a full disk
            logger.warning("could not append metrics: %s", exc)
        self._win_loss, self._win_n, self._win_tokens, self._win_t0 = 0.0, 0, 0, now

    # ------------------------------------------------------------------ main loop
    def run(self) -> LoopResult:
        s, res = self.s, self.res
        now = self.wsd.now()
        last_val = last_ckpt = now
        samples_per_step = s.batch_size * s.grad_accum
        try:
            while res.segment_steps < s.remaining_steps:
                idxs = self._take(samples_per_step)
                if not idxs:
                    logger.warning("[%s] the sampler produced no examples; ending training", self.session.adapter)
                    break
                mbs = self._micro_batches(idxs)
                lr = s.base_lr * self.wsd.lr_lambda(self._global_step)
                step_res = self.t.step(mbs, lr)

                self._global_step += 1
                res.segment_steps += 1
                self._samples_in_epoch += len(idxs)
                stats = [batch_stats(b) for b in mbs]
                step_res["supervised"] = sum(x["supervised"] for x in stats)
                n_tok = sum(x["tokens"] for x in stats)
                res.tokens += n_tok
                res.supervised_tokens += step_res["supervised"]
                self._win_tokens += n_tok

                # Every step: NaN/Inf must not be averaged away by the log window. The update itself was skipped on-device.
                self.session.nonfinite.observe(step_res["loss"], step_res["grad_norm"], self._global_step)
                if not step_res["applied"]:
                    res.skipped_nonfinite_steps += 1
                else:
                    self.session.zerograd.observe(step_res["grad_norm"], step_res["supervised"], self._global_step)
                    self._loss_sum += step_res["loss"]
                    self._loss_n += 1
                    self._win_loss += step_res["loss"]
                    self._win_n += 1

                now = self.wsd.now()
                self.wsd.on_step_end(self._global_step, now=now)

                if (res.segment_steps == 1 or res.segment_steps % max(1, s.log_every) == 0) and self._win_n:
                    self._log(step_res, lr, now)
                if res.segment_steps == s.dead_check_step and res.dead_check_result is None:
                    res.dead_check_result = self.t.dead_lora_report()
                    if res.dead_check_result["dead_lora_b_modules"]:
                        logger.warning("[%s] dead-adapter check: %s", self.session.adapter, res.dead_check_result)
                if res.segment_steps == s.throughput_project_step and res.throughput_projection is None:
                    seg = max(1e-6, now - self.wsd.t_start)
                    tps = res.tokens / seg
                    budget = max(0.0, self.wsd.deadline - self.wsd.t_start)
                    res.throughput_projection = {
                        "measured_at_step": res.segment_steps,
                        "tokens_per_sec": round(tps, 2),
                        "projected_dataset_passes_in_budget": round(tps * budget / max(1.0, float(self.dataset_total_tokens)), 2),
                        "projected_end_offset_sec": round(self.wsd.projected_t_end(now) - self.wsd.t_start, 1),
                    }
                    logger.info("[%s] throughput: %s", self.session.adapter, res.throughput_projection)

                healthy = bool(step_res["applied"])  # no checkpoint / best-adapter bookkeeping in the middle of a NaN streak
                if healthy and self.val_rows and now - last_val >= s.val_every_sec:
                    last_val = now
                    if self._validate(now):
                        res.stopped_early = True
                        break
                if healthy and now - last_ckpt >= s.ckpt_every_sec:
                    last_ckpt = now
                    self.save_checkpoint("wallclock")
                if self.stop_requested is not None and self.stop_requested():
                    logger.warning("[%s] stop requested at step %d", self.session.adapter, self._global_step)
                    res.stopped_by_signal = True
                    break
                if self.wsd.should_stop(now):
                    logger.info("[%s] stopping at step %d (schedule / time budget)", self.session.adapter, self._global_step)
                    break
        except BaseException as exc:
            if type(exc).__name__ in {"NonFiniteLossError", "LossDivergedError", "NoGradientError"}:
                raise  # weights may be corrupt / diverging: never checkpoint them; the restart resumes the last good save
            if s.emergency_checkpoint and res.segment_steps > 0:
                try:
                    saved = self.save_checkpoint("emergency")
                    logger.warning("[%s] emergency checkpoint %s after %s", self.session.adapter, saved, type(exc).__name__)
                except Exception as save_exc:  # noqa: BLE001
                    logger.error("[%s] emergency checkpoint failed: %s", self.session.adapter, save_exc)
            raise
        res.final_train_loss = (self._loss_sum / self._loss_n) if self._loss_n else None
        if res.final_train_loss is not None and not math.isfinite(res.final_train_loss):
            res.final_train_loss = None
        res.epoch = float(self._epoch) + self._samples_in_epoch / max(1, len(self.rows))
        if res.dead_check_result is None:
            res.dead_check_result = self.t.dead_lora_report()
        return res

    def write_final_adapter(self, out_dir: Path) -> None:
        self._write_adapter(out_dir)
