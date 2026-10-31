"""`train:<adapter>` for the Tunix (JAX) backend: same contract as `pipeline/multi_lora_sft/phases/train.py`.

Reads the same Parquet cache, resumes from the same checkpoint layout (this run, `run.resume_from`, attached bundles),
writes `checkpoints/<adapter>/{step_*,final_adapter,best_adapter}`, the same status JSON and manifest summary. The
orchestrator, export phase, packaging and the cross-session bundle cannot tell which backend trained the adapter.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from pipeline.multi_lora_sft.config import (
    DEFAULT_TARGET_REGEX,
    EffectiveAdapterConfig,
    PipelineConfig,
    apply_oom_override,
    load_config,
)
from pipeline.multi_lora_sft.runctx import (
    PhaseStatus,
    RunContext,
    configure_phase_logging,
    utc_iso,
)
from pipeline.multi_lora_sft.shared.fsio import publish_dir, safe_rmtree
from pipeline.multi_lora_sft.shared.session import TrainingSession
from pipeline.multi_lora_sft.training.data import (
    LengthGroupedLongestFirstSampler,
    load_parquet_split,
)
from pipeline.multi_lora_sft.training.schedule import WSDScheduleController

from . import hf_config, model_io, runtime
from .loop import LoopSettings, TrainLoop
from .lora_convert import ConversionError, peft_to_qwix
from .peft_io import OPT_FILE, load_opt_state, qwix_alpha, read_adapter_tensors
from .trainer import LoraTrainer

DEFAULT_PAD_MULTIPLE = 512
_STOP_EVENT = threading.Event()
logger = logging.getLogger(__name__)


def _install_stop_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        _STOP_EVENT.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not the main thread / unsupported platform
            pass


def _write_status(ctx: RunContext, phase_name: str, *, state: str, t_start: float, deadline: float,
                  outputs: dict[str, Any] | None = None, notes: list[str] | None = None, error: str | None = None) -> dict[str, Any]:
    t_end = time.time()
    status = PhaseStatus(
        phase=phase_name, state=state, started=utc_iso(t_start), ended=utc_iso(t_end), started_ts=t_start, ended_ts=t_end,
        planned_deadline=utc_iso(deadline), planned_deadline_ts=deadline, planned_duration_sec=round(deadline - t_start, 2),
        actual_duration_sec=round(t_end - t_start, 2), outputs=outputs or {}, notes=notes or [], error=error,
    )
    ctx.write_status(status)
    return status.to_dict()


def _dtypes(quantization: str, platform: str) -> tuple[Any, Any]:
    import jax.numpy as jnp

    compute = jnp.float32 if (quantization == "fp32" or platform == "cpu") else jnp.bfloat16
    return compute, compute


def run_tunix_adapter_phase(
    ctx: RunContext,
    adapter_name: str,
    *,
    deadline: float,
    cfg: PipelineConfig | None = None,
    eff: EffectiveAdapterConfig | None = None,
    oom_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase_name = f"train:{adapter_name}"
    log = configure_phase_logging(ctx.log_path(phase_name))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)
    if eff is None:
        eff = apply_oom_override(cfg.resolve_adapter(adapter_name), oom_override)
    notes: list[str] = []
    if oom_override:
        notes.append(f"Applied OOM fallback override: {oom_override}")

    try:
        import jax

        _install_stop_handlers()
        platform = runtime.resolve_platform(eff.train.accelerator)
        info = runtime.device_info()
        if info["process_count"] > 1:
            raise runtime.AcceleratorUnavailable(
                f"{info['process_count']} JAX processes: multi-host TPU slices are not supported (use a single-host TPU VM)"
            )
        n_dev = int(info["count"])
        parallelism = eff.train.parallelism
        if parallelism in {"ddp", "model_split"}:
            notes.append(f"parallelism={parallelism!r} is CUDA-only; using 'spmd'")
            parallelism = "spmd"
        if platform == "cpu" or n_dev == 1:
            parallelism = "single"
        log.info("[%s] backend=tunix platform=%s devices=%d parallelism=%s jax=%s", adapter_name, platform, n_dev, parallelism, info["jax"])

        train_base = cfg.model.train_base_path
        if not train_base or not Path(train_base).is_dir():
            raise FileNotFoundError(
                f"model.train_base_path={train_base!r} is not a directory; the Tunix backend loads the bf16 Hugging Face "
                "checkpoint (safetensors) of the model being trained"
            )
        if eff.lora.target_regex != DEFAULT_TARGET_REGEX:
            notes.append("lora.target_regex is ignored by the Tunix backend: LoRA is attached to q/k/v/o and gate/up/down on the language model")
        if eff.lora.dropout:
            notes.append("lora.dropout is not applied by the Tunix backend")

        cache_dir = ctx.cache_dir / adapter_name
        train_rows = load_parquet_split(cache_dir / "train.parquet", max_seq_length=eff.train.max_seq_length)
        val_rows = load_parquet_split(cache_dir / "val.parquet", max_seq_length=eff.train.max_seq_length)
        if not train_rows:
            raise ValueError(f"No training examples available for adapter {adapter_name!r} at max_seq_length={eff.train.max_seq_length}")
        batch_size = eff.train.per_device_batch
        if parallelism == "spmd" and batch_size % n_dev != 0:
            raise ValueError(f"SPMD shards the batch over {n_dev} devices: train.per_device_batch ({batch_size}) must be a multiple of {n_dev}")
        pad_multiple = eff.train.pad_to_multiple_of or (DEFAULT_PAD_MULTIPLE if platform == "tpu" else None)

        # ---- model: load -> LoRA -> remat (this order matters, see model_io.set_remat)
        compute_dtype, param_dtype = _dtypes(eff.model_quantization, platform)
        hf = hf_config.load_text_config(train_base)
        pad_id = int(hf.get("pad_token_id") or 0)
        model_cfg = hf_config.build_model_config(hf, param_dtype=param_dtype, dtype=compute_dtype, remat="NONE")
        mesh = runtime.make_mesh(parallelism)
        random_seed = int(cfg.run.seed)

        with mesh:
            model = model_io.load_base_model(train_base, model_cfg, mesh=mesh, dtype=param_dtype)
            hf_config.check_layout_against_checkpoint(model, train_base)
            parity = model_io.check_forward_parity(model, atol=5e-2 if compute_dtype != jax.numpy.float32 else 2e-3)
            log.info("[%s] training forward matches Tunix Gemma4 (max |diff| %.3g)", adapter_name, parity)
            lora_model = model_io.attach_lora(
                model, rank=eff.lora.r, alpha=qwix_alpha(eff.lora.r, eff.lora.alpha, eff.lora.use_rslora),
                seed=random_seed, dropout=eff.lora.dropout,
            )
            remat = os.environ.get("TUNIX_REMAT") or ("BLOCK" if platform == "tpu" else "NONE")
            model_io.set_remat(lora_model, remat)
            trainer = LoraTrainer(
                lora_model, mesh=mesh, compute_dtype=compute_dtype, chunk_rows=eff.loss.chunk_rows,
                weight_decay=eff.train.weight_decay, max_grad_norm=eff.train.max_grad_norm,
            )
            log.info("[%s] LoRA r=%d alpha=%s: %d trainable parameters, remat=%s", adapter_name, eff.lora.r, eff.lora.alpha, trainer.n_params, remat)

            # ---- schedule, sampler, session (shared with the PyTorch backend)
            samples_per_step = batch_size * eff.train.grad_accum
            steps_per_epoch = max(1, math.ceil(len(train_rows) / samples_per_step))
            total_epoch_steps = max(1, math.ceil(steps_per_epoch * eff.train.max_epochs))
            if eff.train.max_steps is not None:
                total_epoch_steps = min(total_epoch_steps, int(eff.train.max_steps))
            sampler = LengthGroupedLongestFirstSampler(
                [int(r["length"]) for r in train_rows],
                longest_first_samples=eff.train.longest_first_steps * samples_per_step,
                mega_batch_size=eff.train.length_bucket_size, seed=cfg.run.seed,
            )
            raw_window = max(5.0, deadline - t_start)
            soft_deadline = t_start + max(2.0, raw_window - min(30.0, raw_window * 0.12))
            wsd = WSDScheduleController(
                warmup_steps=eff.schedule.warmup_steps, decay_frac=eff.schedule.decay_frac,
                final_lr_ratio=eff.schedule.final_lr_ratio, ema_beta=eff.schedule.step_time_ema_beta,
                total_epoch_steps=total_epoch_steps, t_start=t_start, deadline=soft_deadline,
            )
            manifest = ctx.read_manifest()
            session = TrainingSession(
                run_checkpoints_dir=ctx.checkpoints_dir, run_dir=ctx.run_dir, cfg=cfg, eff=eff, adapter_name=adapter_name,
                backend="tunix", accelerator=platform, is_main=True, train_rows=train_rows,
                tokenizer_sha=manifest.get("tokenizer_sha256"), template_sha=manifest.get("template_sha256"),
                train_base_name="/".join(Path(train_base).parts[-2:]) or None,
            )

            # ---- resume: weights first; only a checkpoint that actually loads may skip data
            resume_info = session.resolve_resume()
            resume_plan = None
            resumed_step = 0
            if resume_info is not None:
                try:
                    arrays = peft_to_qwix(read_adapter_tensors(resume_info.checkpoint / "adapter"), trainer.template)
                    trainer.load_lora_numpy(arrays)
                except (ConversionError, model_io.ModelLoadError, OSError, ValueError) as exc:
                    msg = f"checkpoint {resume_info.checkpoint} cannot be loaded into the Tunix model ({exc}); starting a fresh LoRA"
                    log.warning("[%s] %s", adapter_name, msg)
                    notes.append(msg)
                    session.notes.append(msg)
                    resume_info = None
            if resume_info is not None:
                meta = resume_info.meta
                resumed_step = int(meta.get("step", 0))
                resume_plan = session.plan_data(resume_info)
                if resume_plan is not None:
                    if resume_plan.mode in {"continue", "next_epoch"}:
                        sampler.set_exclusion(resume_plan.exclude_indices, resume_plan.epoch)
                    notes.append(f"Data resume: {resume_plan.note}")
                elif meta.get("sampler_state"):
                    sampler.load_state_dict(meta["sampler_state"])
                if meta.get("schedule_state"):
                    wsd.load_resume_state(meta["schedule_state"], new_t_start=t_start, new_deadline=soft_deadline, extend=cfg.resume.extend)
                if resume_info.optimizer_compatible:
                    leaves = load_opt_state(resume_info.checkpoint / OPT_FILE, trainer.opt_shapes())
                    if leaves is not None:
                        trainer.load_opt_leaves(leaves)
                    else:
                        notes.append("Optimizer state not restored (fresh Adam moments)")
                else:
                    notes.append("Optimizer state not reused (checkpoint written by a different backend / settings)")
                notes.append(f"Resumed from {resume_info.checkpoint} at step {resumed_step}")
                log.info("[%s] restored %s (step %d)", adapter_name, resume_info.checkpoint, resumed_step)

            already_done = resume_info is not None and resumed_step >= total_epoch_steps and not cfg.resume.extend
            remaining = max(1, total_epoch_steps - resumed_step)

            ckpt_root = ctx.checkpoints_dir / adapter_name
            ckpt_root.mkdir(parents=True, exist_ok=True)
            settings = LoopSettings(
                batch_size=batch_size, grad_accum=eff.train.grad_accum, base_lr=eff.train.lr, pad_id=pad_id,
                pad_multiple=pad_multiple, remaining_steps=remaining, step_offset=resumed_step, r=eff.lora.r,
                alpha=float(eff.lora.alpha), use_rslora=bool(eff.lora.use_rslora),
                base_model_id=cfg.effective_served_model_id(), log_every=eff.train.log_every_steps,
                val_every_sec=eff.train.val_every_min * 60.0, ckpt_every_sec=eff.train.checkpoint_every_min * 60.0,
                keep_checkpoints=eff.train.keep_checkpoints, min_free_disk_gib=eff.train.min_free_disk_gib,
                dead_check_step=eff.train.dead_check_step, throughput_project_step=eff.train.throughput_project_step,
                emergency_checkpoint=eff.train.emergency_checkpoint,
            )
            loop = TrainLoop(
                trainer, session, wsd, sampler, train_rows, val_rows, settings, ckpt_root=ckpt_root,
                metrics_path=ctx.metrics_dir / f"{adapter_name}.jsonl", stop_requested=_STOP_EVENT.is_set,
                dataset_total_tokens=sum(int(r["length"]) for r in train_rows),
            )
            session.start_tracking({
                "fingerprint": session.fingerprint, "data": session.data_fp, "train": dataclasses.asdict(eff.train),
                "lora": dataclasses.asdict(eff.lora), "resumed_step": resumed_step, "backend": "tunix", "devices": n_dev,
            })

            if already_done:
                notes.append(
                    f"Checkpoint already at step {resumed_step} >= target {total_epoch_steps}; exporting it without further "
                    "training (set resume.extend=true to keep training)"
                )
                log.info("[%s] %s", adapter_name, notes[-1])
                result = loop.res
                result.dead_check_result = trainer.dead_lora_report()
            else:
                log.info("[%s] training: %d examples, %d steps, batch %d x accum %d, window %.0fs",
                         adapter_name, len(train_rows), remaining, batch_size, eff.train.grad_accum, raw_window)
                result = loop.run()
                if result.stopped_by_signal:
                    notes.append("Stopped early by SIGTERM/SIGINT; final adapter saved at the last completed step")
                if result.stopped_early:
                    notes.append("Stopped by validation early stopping")
                if result.skipped_nonfinite_steps:
                    notes.append(f"{result.skipped_nonfinite_steps} step(s) skipped for non-finite gradients")

            final_val_loss = None
            if val_rows and not _STOP_EVENT.is_set() and (deadline - time.time()) > 5.0:
                v = loop.evaluate()
                final_val_loss = round(float(v), 6) if v is not None else None

            final_dir = ckpt_root / "final_adapter"
            tmp_final = ckpt_root / ".tmp_final_adapter"
            safe_rmtree(tmp_final)
            loop.write_final_adapter(tmp_final)
            publish_dir(tmp_final, final_dir)

            steps_completed = resumed_step + result.segment_steps
            final_train_loss = round(result.final_train_loss, 6) if result.final_train_loss is not None else None
            session_out = session.finalize(
                final_dir, steps_completed=steps_completed,
                summary={"final_val_loss": final_val_loss, "final_train_loss": final_train_loss, "steps_completed": steps_completed},
            )
            if not wsd.ever_decayed and eff.schedule.decay_frac > 0.0 and steps_completed > 1 and not already_done:
                notes.append("Training ended before entering WSD decay phase")

            summary = {
                "adapter": adapter_name, "accelerator": platform, "parallelism": parallelism, "loss_impl": "chunked",
                "steps_completed": steps_completed, "segment_steps": result.segment_steps, "target_steps": total_epoch_steps,
                "epochs_completed": round(result.epoch, 4), "final_train_loss": final_train_loss, "final_val_loss": final_val_loss,
                "decay_entered": wsd.ever_decayed, "final_lr_multiplier": round(float(wsd.last_multiplier), 6),
                "dead_lora_check": result.dead_check_result, "throughput_projection": result.throughput_projection,
                "matched_module_count": len(json.loads((final_dir / "matched_target_modules.json").read_text())["matched_modules"]),
                "final_adapter_dir": str(final_dir), "parity_hf_path": None, "oom_override": oom_override,
                "resumed_from": str(resume_info.checkpoint) if resume_info else None, "stopped_by_signal": result.stopped_by_signal,
                "backend": "tunix", "device": info, "tokens_trained": result.tokens, "session": session_out,
                "resume": {
                    "checkpoint": str(resume_info.checkpoint) if resume_info else None,
                    "optimizer_restored": bool(resume_info and resume_info.optimizer_compatible),
                    "data_plan": resume_plan.note if resume_plan else None, "notes": session.notes,
                },
            }
            ctx.update_manifest({"training_summaries": {**ctx.read_manifest().get("training_summaries", {}), adapter_name: summary}})
            out = _write_status(ctx, phase_name, state="ok", t_start=t_start, deadline=deadline, outputs=summary, notes=notes)
            log.info("[%s] finished %d step(s) in %.1fs (train_loss=%s, val_loss=%s)", adapter_name, steps_completed,
                     time.time() - t_start, final_train_loss, final_val_loss)
            del trainer, lora_model, model
        jax.clear_caches()
        return out
    except BaseException as exc:
        log.error("[%s] Training failed: %s: %s", adapter_name, type(exc).__name__, exc)
        _write_status(ctx, phase_name, state="failed", t_start=t_start, deadline=deadline, notes=notes,
                      error=f"{type(exc).__name__}: {exc}")
        raise
