"""Phase 3 (`train:<adapter>`): isolated subprocess training of a single LoRA adapter.

Implements design_doc.md §6.2 and §7.3–§7.10:
- Loads pre-tokenized Parquet cache from `runs/<run_id>/cache/<adapter>/`.
- Applies optional OOM fallback overrides (`--oom-override-json`).
- Restores mid-run crash or cross-session (`run.resume_from`) checkpoints including LoRA weights,
  optimizer, scheduler, RNG states, sampler offset, and WSD schedule state (§7.8). Unreadable checkpoints are
  skipped in favour of the next-newest one, and a checkpoint that already reached the step target is exported
  without retraining.
- Trains via `SelectiveLossSFTTrainer` with softcapped selective-logits CE (or the static-shape variant on TPU),
  `LengthGroupedLongestFirstSampler`, `WSDScheduleController`, and `MultiLoraTrainingCallback`.
- Runs on CUDA (single / DDP / model_split), TPU via torch_xla (single core or SPMD FSDPv2), or CPU.
- SIGTERM (orchestrator hard deadline) stops training at the next step boundary and still saves the adapter;
  a crash mid-run writes an emergency checkpoint first (never on NaN/Inf, which aborts to the last good one).
- Evaluates final validation loss, writes `checkpoints/<adapter>/final_adapter/`, and generates
  HF greedy parity reference JSON (`parity/<adapter>_hf.json`, CUDA/CPU only) before freeing device memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Any

from ..config import EffectiveAdapterConfig, PipelineConfig, load_config
from ..runctx import PhaseStatus, RunContext, configure_phase_logging, utc_iso
from ..shared.backend import resolve_backend
from ..shared.bundle import discover_bundles, resume_candidates as shared_resume_candidates
from ..shared.session import TrainingSession
from ..training.callbacks import (
    MultiLoraTrainingCallback,
    NonFiniteLossError,
    checkpoint_is_loadable,
    save_adapter_weights,
)
from ..training.data import (
    LengthGroupedLongestFirstSampler,
    load_parquet_split,
    load_served_tokenizer_and_template,
)
from ..training.device import (
    empty_cache,
    enable_spmd,
    reset_peak_memory,
    resolve_accelerator,
    restore_rng_state,
    xla_device_count,
)
from ..training.loss import (
    WeightedLanguageModelingCollator,
    build_selective_sft_trainer_cls,
)
from ..training.model import attach_lora_adapter, load_base_model_for_training
from ..training.parity import generate_hf_parity_reference
from ..training.schedule import WSDScheduleController, create_wsd_scheduler

TPU_DEFAULT_PAD_MULTIPLE = 512
GEMMA4_DECODER_LAYER_CLS = "Gemma4TextDecoderLayer"

_STOP_EVENT = threading.Event()


def _install_stop_handlers() -> None:
    """SIGTERM/SIGINT only set a flag; the training callback stops at the next step and weights are saved."""

    def _handler(signum: int, _frame: Any) -> None:
        _STOP_EVENT.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # not on the main thread / unsupported on this platform
            pass


def apply_oom_override(eff: EffectiveAdapterConfig, override: dict[str, Any] | None) -> EffectiveAdapterConfig:
    """Applies a single `oom_fallbacks` dict onto `EffectiveAdapterConfig`."""
    if not override:
        return eff
    train_dict = dataclasses.asdict(eff.train)
    quant = eff.model_quantization
    for k, v in override.items():
        if k == "quantization":
            quant = str(v)
        elif k in train_dict:
            train_dict[k] = v
    return EffectiveAdapterConfig(
        name=eff.name,
        roles=eff.roles,
        time_share=eff.time_share,
        lora=eff.lora,
        train=type(eff.train)(**train_dict),
        schedule=eff.schedule,
        loss=eff.loss,
        model_quantization=quant,
    )


def resume_candidates(ctx: RunContext, cfg: PipelineConfig, adapter_name: str) -> list[Path]:
    """Resumable `step_<n>/` checkpoints, newest first: this run's (crash restart), `run.resume_from`, attached bundles."""
    bundles = [Path(p) for p in cfg.resume.bundle_dirs]
    if cfg.resume.auto_discover:
        bundles += [b for b in discover_bundles([cfg.resume.input_root]) if b not in bundles]
    return shared_resume_candidates(ctx.checkpoints_dir, adapter_name, resume_from=cfg.run.resume_from, bundle_dirs=bundles)


def resolve_resume_checkpoint(ctx: RunContext, cfg: PipelineConfig, adapter_name: str) -> Path | None:
    """Newest checkpoint whose metadata and adapter weights are readable (corrupt ones are skipped)."""
    for cand in resume_candidates(ctx, cfg, adapter_name):
        if checkpoint_is_loadable(cand):
            return cand
    return None


def _load_torch(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _write_status(
    ctx: RunContext,
    phase_name: str,
    *,
    state: str,
    t_start: float,
    deadline: float,
    outputs: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    t_end = time.time()
    status = PhaseStatus(
        phase=phase_name,
        state=state,
        started=utc_iso(t_start),
        ended=utc_iso(t_end),
        started_ts=t_start,
        ended_ts=t_end,
        planned_deadline=utc_iso(deadline),
        planned_deadline_ts=deadline,
        planned_duration_sec=round(deadline - t_start, 2),
        actual_duration_sec=round(t_end - t_start, 2),
        outputs=outputs or {},
        notes=notes or [],
        error=error,
    )
    ctx.write_status(status)
    return status.to_dict()


def run_train_adapter_phase(
    ctx: RunContext,
    adapter_name: str,
    *,
    deadline: float,
    cfg: PipelineConfig | None = None,
    oom_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trains a single adapter inside the current process and writes `status/train__<adapter>.json`."""
    phase_name = f"train:{adapter_name}"
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    logger = configure_phase_logging(ctx.log_path(phase_name) if is_main else None)

    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)
    eff = apply_oom_override(cfg.resolve_adapter(adapter_name), oom_override)

    backend = resolve_backend(
        eff.train.backend, served_model_id=cfg.effective_served_model_id(), accelerator=eff.train.accelerator
    )
    if backend == "tunix":  # JAX trainer for E4B/E2B on TPU: same session, same checkpoints, same adapter format
        from tpu_training.phase import run_tunix_adapter_phase

        logger.info("[%s] backend=tunix (train.backend=%s)", adapter_name, eff.train.backend)
        return run_tunix_adapter_phase(ctx, adapter_name, deadline=deadline, cfg=cfg, eff=eff, oom_override=oom_override)

    notes: list[str] = []
    if oom_override:
        notes.append(f"Applied OOM fallback override: {oom_override}")

    cb: MultiLoraTrainingCallback | None = None
    trainer: Any = None

    try:
        import datasets as hf_datasets
        import numpy as np
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors
        from trl import SFTConfig

        accelerator = resolve_accelerator(eff.train.accelerator)
        parallelism = eff.train.parallelism
        if accelerator == "tpu" and parallelism in {"ddp", "model_split"}:
            notes.append(f"parallelism={parallelism!r} is CUDA-only; using 'spmd' on TPU")
            parallelism = "spmd"
        if accelerator != "tpu" and parallelism == "spmd":
            notes.append(f"parallelism='spmd' needs a TPU; using 'single' on {accelerator}")
            parallelism = "single"
        if parallelism != eff.train.parallelism:
            eff = dataclasses.replace(eff, train=dataclasses.replace(eff.train, parallelism=parallelism))
        is_tpu = accelerator == "tpu"
        if is_tpu and parallelism == "spmd":
            enable_spmd()  # must precede any XLA tensor creation

        loss_impl = eff.loss.impl
        if is_tpu and loss_impl != "static":
            notes.append(f"loss.impl={loss_impl!r} has data-dependent shapes; using 'static' on TPU")
            loss_impl = "static"
        pad_multiple = eff.train.pad_to_multiple_of or (TPU_DEFAULT_PAD_MULTIPLE if is_tpu else None)
        logger.info(
            "[%s] accelerator=%s parallelism=%s loss=%s pad_to_multiple_of=%s",
            adapter_name,
            accelerator,
            parallelism,
            loss_impl,
            pad_multiple,
        )

        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.manual_seed(cfg.run.seed)
        if accelerator == "cuda":
            torch.cuda.manual_seed_all(cfg.run.seed)
        reset_peak_memory(accelerator)
        _install_stop_handlers()

        cache_dir = ctx.cache_dir / adapter_name
        train_rows = load_parquet_split(cache_dir / "train.parquet", max_seq_length=eff.train.max_seq_length)
        val_rows = load_parquet_split(cache_dir / "val.parquet", max_seq_length=eff.train.max_seq_length)
        if not train_rows:
            raise ValueError(
                f"No training examples available for adapter {adapter_name!r} at max_seq_length={eff.train.max_seq_length}"
            )

        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        if is_tpu and parallelism == "spmd":
            n_dev = xla_device_count()
            if eff.train.per_device_batch % n_dev != 0:
                raise ValueError(
                    f"SPMD shards the batch over {n_dev} TPU devices: train.per_device_batch "
                    f"({eff.train.per_device_batch}) must be a multiple of {n_dev}"
                )
            if len(train_rows) < eff.train.per_device_batch:
                raise ValueError(
                    f"Only {len(train_rows)} training rows but SPMD needs full batches of {eff.train.per_device_batch}"
                )

        train_ds = hf_datasets.Dataset.from_list(train_rows)
        val_ds = hf_datasets.Dataset.from_list(val_rows) if val_rows else None

        tiny_dir = ctx.cache_dir / "_tiny_gemma4"
        tokenizer, _, _, _ = load_served_tokenizer_and_template(cfg, tiny_fallback_dir=tiny_dir)
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; cannot pad batches")

        base_model = load_base_model_for_training(cfg, eff, tiny_fallback_dir=tiny_dir)
        base_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id

        peft_model, target_regex, matched_modules = attach_lora_adapter(base_model, eff.lora)
        if is_tpu and parallelism == "single":
            import torch_xla.core.xla_model as xm

            peft_model = peft_model.to(xm.xla_device())

        samples_per_step = eff.train.per_device_batch * eff.train.grad_accum * world_size
        steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, eff.train.per_device_batch * world_size) / eff.train.grad_accum))
        total_epoch_steps = max(1, math.ceil(steps_per_epoch * eff.train.max_epochs))
        if eff.train.max_steps is not None:
            total_epoch_steps = min(total_epoch_steps, int(eff.train.max_steps))

        longest_first_samples = eff.train.longest_first_steps * samples_per_step
        sampler = LengthGroupedLongestFirstSampler(
            [int(r["length"]) for r in train_rows],
            longest_first_samples=longest_first_samples,
            mega_batch_size=eff.train.length_bucket_size,
            seed=cfg.run.seed,
        )

        # Reserve ~30s before hard phase kill so final save + parity reference finish cleanly
        raw_window = max(5.0, deadline - t_start)
        soft_guard = min(30.0, raw_window * 0.12)
        soft_deadline = t_start + max(2.0, raw_window - soft_guard)

        wsd = WSDScheduleController(
            warmup_steps=eff.schedule.warmup_steps,
            decay_frac=eff.schedule.decay_frac,
            final_lr_ratio=eff.schedule.final_lr_ratio,
            ema_beta=eff.schedule.step_time_ema_beta,
            total_epoch_steps=total_epoch_steps,
            t_start=t_start,
            deadline=soft_deadline,
        )

        # Shared session: resume compatibility, task ledger, guards, best-checkpoint selection, tracking.
        session = TrainingSession(
            run_checkpoints_dir=ctx.checkpoints_dir,
            run_dir=ctx.run_dir,
            cfg=cfg,
            eff=eff,
            adapter_name=adapter_name,
            backend="trl",
            accelerator=accelerator,
            is_main=is_main,
            train_rows=train_rows,
            tokenizer_sha=ctx.read_manifest().get("tokenizer_sha256"),
            template_sha=ctx.read_manifest().get("template_sha256"),
            train_base_name="/".join(Path(cfg.model.train_base_path or "").parts[-2:]) or None,
        )

        # Resume checkpoint restoration (§7.8)
        resume_info = session.resolve_resume()
        resume_ckpt = resume_info.checkpoint if resume_info else None
        resume_plan = session.plan_data(resume_info)
        resumed_step = 0
        resume_meta: dict[str, Any] = {}
        if resume_ckpt is not None:
            resume_meta = resume_info.meta
            resumed_step = int(resume_meta.get("step", 0))
            adapter_st = load_safetensors(str(resume_ckpt / "adapter" / "adapter_model.safetensors"), device="cpu")
            load_result = set_peft_model_state_dict(peft_model, adapter_st)
            unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
            if unexpected:
                raise ValueError(
                    f"Checkpoint {resume_ckpt} does not match this LoRA config ({len(unexpected)} unexpected keys, "
                    f"first {unexpected[:3]}); was lora.r / target_regex changed since it was written?"
                )
            if resume_plan is not None:  # task-level resume: skip what was trained on, whatever the data looks like now
                if resume_plan.mode in {"continue", "next_epoch"}:
                    sampler.set_exclusion(resume_plan.exclude_indices, resume_plan.epoch)
                notes.append(f"Data resume: {resume_plan.note}")
            elif resume_meta.get("sampler_state"):
                sampler.load_state_dict(resume_meta["sampler_state"])
            if resume_meta.get("schedule_state"):
                wsd.load_resume_state(
                    resume_meta["schedule_state"],
                    new_t_start=t_start,
                    new_deadline=soft_deadline,
                    extend=cfg.resume.extend,
                )
            rng_path = resume_ckpt / "rng_state.pt"
            if rng_path.is_file():
                restore_rng_state(_load_torch(rng_path), accelerator)
            notes.append(f"Resumed from {resume_ckpt} at step {resumed_step}")
            logger.info("[%s] Restored checkpoint %s (step=%d)", adapter_name, resume_ckpt, resumed_step)

        already_done = resume_ckpt is not None and resumed_step >= total_epoch_steps and not cfg.resume.extend
        remaining_steps = max(1, total_epoch_steps - resumed_step)

        def _build_optimizer_and_scheduler(model_for_opt: Any) -> tuple[Any, Any]:
            params = [p for p in model_for_opt.parameters() if p.requires_grad]
            if not params:
                raise ValueError("No trainable parameters after attaching LoRA (all adapters frozen?)")
            opt = torch.optim.AdamW(params, lr=eff.train.lr, weight_decay=eff.train.weight_decay)
            if resume_ckpt is not None and (resume_ckpt / "optimizer.pt").is_file() and resume_info.optimizer_compatible:
                try:
                    opt.load_state_dict(_load_torch(resume_ckpt / "optimizer.pt"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not restore optimizer state (%s); continuing with fresh moments", exc)
                    notes.append("Optimizer state not restored (fresh Adam moments)")
            sched = create_wsd_scheduler(opt, wsd)
            if resume_ckpt is not None and (resume_ckpt / "scheduler.pt").is_file():
                try:
                    sched.load_state_dict(_load_torch(resume_ckpt / "scheduler.pt"))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not restore scheduler state: %s", exc)
            return opt, sched

        use_fsdp_xla = is_tpu and parallelism == "spmd"
        optimizers: tuple[Any, Any] = (None, None) if use_fsdp_xla else _build_optimizer_and_scheduler(peft_model)

        # bf16 autocast on CUDA; on TPU the weights are already bf16 and XLA handles mixed precision itself.
        use_bf16 = eff.model_quantization in {"bf16", "nf4"} and accelerator == "cuda"

        ckpt_root = ctx.checkpoints_dir / adapter_name
        ckpt_root.mkdir(parents=True, exist_ok=True)

        fsdp_kwargs: dict[str, Any] = {}
        if use_fsdp_xla:
            fsdp_kwargs = {
                "fsdp": "full_shard",
                "fsdp_config": {
                    "xla": True,
                    "xla_fsdp_v2": True,
                    "xla_fsdp_grad_ckpt": True,
                    "transformer_layer_cls_to_wrap": [GEMMA4_DECODER_LAYER_CLS],
                },
            }

        sft_args = SFTConfig(
            output_dir=str(ckpt_root / "hf_trainer"),
            num_train_epochs=float(eff.train.max_epochs),
            max_steps=remaining_steps,
            per_device_train_batch_size=eff.train.per_device_batch,
            per_device_eval_batch_size=eff.train.per_device_batch if use_fsdp_xla else 1,
            gradient_accumulation_steps=eff.train.grad_accum,
            learning_rate=eff.train.lr,
            weight_decay=eff.train.weight_decay,
            max_grad_norm=eff.train.max_grad_norm,
            bf16=use_bf16,
            fp16=False,
            logging_steps=eff.train.log_every_steps,
            logging_first_step=True,
            logging_nan_inf_filter=False,  # NaN must reach the callback's guard, not be averaged away
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=cfg.run.seed,
            dataloader_num_workers=0,
            dataloader_drop_last=is_tpu,  # static batch dimension on XLA
            remove_unused_columns=True,
            average_tokens_across_devices=True,
            ddp_find_unused_parameters=(parallelism == "ddp" and not cfg.model.skip_multimodal_towers),
            gradient_checkpointing=not use_fsdp_xla,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=None,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            **fsdp_kwargs,
        )

        metrics_tracker: dict[str, float] = {}
        dataset_total_tokens = sum(int(r["length"]) for r in train_rows)
        cb = MultiLoraTrainingCallback(
            adapter_name=adapter_name,
            wsd_controller=wsd,
            metrics_jsonl_path=ctx.metrics_dir / f"{adapter_name}.jsonl",
            checkpoint_root=ckpt_root,
            metrics_tracker=metrics_tracker,
            sampler=sampler,
            dataset_total_tokens=dataset_total_tokens,
            samples_per_step=samples_per_step,
            log_every_steps=eff.train.log_every_steps,
            val_every_min=eff.train.val_every_min,
            checkpoint_every_min=eff.train.checkpoint_every_min,
            keep_checkpoints=eff.train.keep_checkpoints,
            dead_check_step=eff.train.dead_check_step,
            throughput_project_step=eff.train.throughput_project_step,
            has_eval_dataset=(val_ds is not None),
            step_offset=resumed_step,
            accelerator=accelerator,
            min_free_disk_gib=eff.train.min_free_disk_gib,
            max_nonfinite_logs=eff.train.max_nonfinite_logs,
            stop_requested=_STOP_EVENT.is_set,
            session=session,
        )

        collator = WeightedLanguageModelingCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=pad_multiple)
        trainer_cls = build_selective_sft_trainer_cls()
        trainer = trainer_cls(
            model=peft_model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=collator,
            optimizers=optimizers,
            callbacks=[cb],
            chunk_rows=eff.loss.chunk_rows,
            loss_impl=loss_impl,
            checkpoint_chunks=eff.loss.checkpoint_chunks,
            custom_train_sampler=sampler,
            metrics_tracker=metrics_tracker,
            optimizer_factory=_build_optimizer_and_scheduler if use_fsdp_xla else None,
            tuple_outputs=use_fsdp_xla,
        )
        cb.bind_trainer(trainer)
        session.start_tracking(
            {"fingerprint": session.fingerprint, "data": session.data_fp, "train": dataclasses.asdict(eff.train),
             "lora": dataclasses.asdict(eff.lora), "resumed_step": resumed_step}
        )

        segment_steps = 0
        final_train_loss: float | None = None
        if already_done:
            notes.append(
                f"Checkpoint already at step {resumed_step} >= target {total_epoch_steps}; exporting it without "
                "further training (set resume.extend=true to keep training)"
            )
            logger.info("[%s] %s", adapter_name, notes[-1])
        else:
            logger.info(
                "[%s] Starting training: %d examples, r=%d, alpha=%d, matched_modules=%d, max_steps=%d, window=%.1fs",
                adapter_name,
                len(train_rows),
                eff.lora.r,
                eff.lora.alpha,
                len(matched_modules),
                remaining_steps,
                raw_window,
            )
            try:
                train_result = trainer.train()
            except NonFiniteLossError:
                raise  # weights may be corrupt: never checkpoint them; the restart resumes the last good save
            except BaseException as exc:
                if eff.train.emergency_checkpoint and trainer.state.global_step > 0:
                    try:
                        empty_cache(accelerator)
                        saved = cb.save_checkpoint(trainer.state, trainer.model, reason="emergency")
                        if saved is not None:
                            logger.warning("[%s] Emergency checkpoint written to %s after %s", adapter_name, saved, type(exc).__name__)
                    except Exception as save_exc:  # noqa: BLE001
                        logger.error("[%s] Emergency checkpoint failed: %s", adapter_name, save_exc)
                raise
            segment_steps = int(trainer.state.global_step)
            final_train_loss = round(float(train_result.training_loss), 6)
            if not math.isfinite(final_train_loss):
                raise NonFiniteLossError(f"[{adapter_name}] final training loss is {final_train_loss}")
            cb.finalize_checks(trainer.model)
            if cb.stopped_by_signal:
                notes.append("Stopped early by SIGTERM/SIGINT; final adapter saved at the last completed step")

        final_val_loss: float | None = None
        if val_ds is not None and not _STOP_EVENT.is_set() and (deadline - time.time()) > 5.0:
            eval_metrics = trainer.evaluate()
            if "eval_loss" in eval_metrics:
                final_val_loss = round(float(eval_metrics["eval_loss"]), 6)

        final_adapter_dir = ckpt_root / "final_adapter"
        parity_path: Path | None = None
        if is_main:
            final_adapter_dir.mkdir(parents=True, exist_ok=True)
            save_adapter_weights(trainer.model, final_adapter_dir, accelerator)
            (final_adapter_dir / "matched_target_modules.json").write_text(
                json.dumps(
                    {
                        "target_regex": target_regex,
                        "matched_modules": matched_modules,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if is_tpu and cfg.parity.enabled:
                notes.append("HF parity reference skipped on TPU (token-by-token generate recompiles every step)")
            elif cfg.parity.enabled and not _STOP_EVENT.is_set() and (deadline - time.time()) > 3.0:
                unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                gen_toks = min(cfg.parity.gen_tokens, 16 if cfg.run.mode == "calibration" else cfg.parity.gen_tokens)
                n_prompts = min(cfg.parity.num_prompts, 2 if cfg.run.mode == "calibration" else cfg.parity.num_prompts)
                parity_path = generate_hf_parity_reference(
                    adapter_name=adapter_name,
                    peft_model=unwrapped,
                    tokenizer=tokenizer,
                    val_records=val_rows,
                    train_records=train_rows,
                    output_path=ctx.parity_dir / f"{adapter_name}_hf.json",
                    num_prompts=n_prompts,
                    max_prompt_tokens=cfg.parity.max_prompt_tokens,
                    gen_tokens=gen_toks,
                )

        steps_completed = resumed_step + segment_steps
        session_out: dict[str, Any] = {}
        if is_main:
            session_out = session.finalize(
                final_adapter_dir,
                steps_completed=steps_completed,
                summary={"final_val_loss": final_val_loss, "final_train_loss": final_train_loss, "steps_completed": steps_completed},
            )
        if not wsd.ever_decayed and eff.schedule.decay_frac > 0.0 and steps_completed > 1 and not already_done:
            notes.append("Training ended before entering WSD decay phase")

        summary = {
            "adapter": adapter_name,
            "accelerator": accelerator,
            "parallelism": parallelism,
            "loss_impl": loss_impl,
            "steps_completed": steps_completed,
            "segment_steps": segment_steps,
            "target_steps": total_epoch_steps,
            "epochs_completed": round(float(trainer.state.epoch or 0.0), 4),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "decay_entered": wsd.ever_decayed,
            "final_lr_multiplier": round(float(wsd.last_multiplier), 6),
            "dead_lora_check": cb.dead_check_result,
            "throughput_projection": cb.throughput_projection,
            "matched_module_count": len(matched_modules),
            "final_adapter_dir": str(final_adapter_dir),
            "parity_hf_path": str(parity_path) if parity_path else None,
            "oom_override": oom_override,
            "resumed_from": str(resume_ckpt) if resume_ckpt else None,
            "stopped_by_signal": cb.stopped_by_signal,
            "backend": "trl",
            "session": session_out,
            "resume": {
                "checkpoint": str(resume_ckpt) if resume_ckpt else None,
                "optimizer_restored": bool(resume_info and resume_info.optimizer_compatible),
                "data_plan": resume_plan.note if resume_plan else None,
                "notes": session.notes,
            },
        }

        if is_main:
            manifest = ctx.read_manifest()
            tr_map = dict(manifest.get("training_summaries", {}))
            tr_map[adapter_name] = summary
            ctx.update_manifest({"training_summaries": tr_map})
            result_dict = _write_status(
                ctx, phase_name, state="ok", t_start=t_start, deadline=deadline, outputs=summary, notes=notes
            )
            logger.info(
                "[%s] Finished %d step(s) in %.2fs (train_loss=%s, val_loss=%s)",
                adapter_name,
                steps_completed,
                time.time() - t_start,
                final_train_loss,
                final_val_loss,
            )
        else:
            result_dict = {"phase": phase_name, "state": "ok", "outputs": summary}

        del trainer, peft_model, base_model
        empty_cache(accelerator)
        return result_dict

    except BaseException as exc:
        if is_main:
            logger.error("[%s] Training failed: %s: %s", adapter_name, type(exc).__name__, exc)
            _write_status(
                ctx,
                phase_name,
                state="failed",
                t_start=t_start,
                deadline=deadline,
                notes=notes,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 3: train single adapter")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--oom-override-json", default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    oom_ov = json.loads(args.oom_override_json) if args.oom_override_json else None
    try:
        run_train_adapter_phase(ctx, args.adapter, deadline=args.deadline, oom_override=oom_ov)
        return 0
    except BaseException:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
