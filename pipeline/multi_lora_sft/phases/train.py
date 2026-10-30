"""Phase 3 (`train:<adapter>`): isolated subprocess training of a single LoRA adapter.

Implements design_doc.md §6.2 and §7.3–§7.10:
- Loads pre-tokenized Parquet cache from `runs/<run_id>/cache/<adapter>/`.
- Applies optional OOM fallback overrides (`--oom-override-json`).
- Restores mid-run crash or cross-session (`run.resume_from`) checkpoints including LoRA weights,
  optimizer, scheduler, RNG states, sampler offset, and WSD schedule state (§7.8).
- Trains via `SelectiveLossSFTTrainer` with softcapped selective-logits CE, `LengthGroupedLongestFirstSampler`,
  `WSDScheduleController`, and `MultiLoraTrainingCallback`.
- Evaluates final validation loss, writes `checkpoints/<adapter>/final_adapter/`, and generates
  HF greedy parity reference JSON (`parity/<adapter>_hf.json`) before freeing GPU memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from ..config import EffectiveAdapterConfig, PipelineConfig, load_config
from ..runctx import PhaseStatus, RunContext, configure_phase_logging, utc_iso
from ..training.callbacks import MultiLoraTrainingCallback, find_latest_checkpoint
from ..training.data import (
    LengthGroupedLongestFirstSampler,
    load_parquet_split,
    load_served_tokenizer_and_template,
)
from ..training.loss import (
    WeightedLanguageModelingCollator,
    build_selective_sft_trainer_cls,
)
from ..training.model import attach_lora_adapter, load_base_model_for_training
from ..training.parity import generate_hf_parity_reference
from ..training.schedule import WSDScheduleController, create_wsd_scheduler


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


def resolve_resume_checkpoint(ctx: RunContext, cfg: PipelineConfig, adapter_name: str) -> Path | None:
    """Finds a resumable `step_<n>/` checkpoint from current run (crash restart) or `run.resume_from`."""
    local_latest = find_latest_checkpoint(ctx.checkpoints_dir / adapter_name)
    if local_latest is not None:
        return local_latest

    if cfg.run.resume_from:
        rp = Path(cfg.run.resume_from)
        # Support pointing directly to a step_* dir, an adapter checkpoint root, or a prior run root
        if (rp / "trainer_resume.json").is_file():
            return rp
        cand1 = find_latest_checkpoint(rp / "checkpoints" / adapter_name)
        if cand1 is not None:
            return cand1
        cand2 = find_latest_checkpoint(rp / adapter_name)
        if cand2 is not None:
            return cand2
        cand3 = find_latest_checkpoint(rp)
        if cand3 is not None:
            return cand3
    return None


def _restore_rng_states(rng_path: Path) -> None:
    import numpy as np
    import torch

    if not rng_path.is_file():
        return
    saved = torch.load(rng_path, map_location="cpu", weights_only=False)
    if "python" in saved:
        random.setstate(saved["python"])
    if "numpy" in saved:
        np.random.set_state(saved["numpy"])
    if "torch_cpu" in saved:
        torch.set_rng_state(saved["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in saved:
        try:
            torch.cuda.set_rng_state_all(saved["torch_cuda"])
        except (RuntimeError, ValueError, TypeError):
            pass


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

    notes: list[str] = []
    if oom_override:
        notes.append(f"Applied OOM fallback override: {oom_override}")

    try:
        import datasets as hf_datasets
        import numpy as np
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors
        from trl import SFTConfig

        # Seed deterministically
        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.manual_seed(cfg.run.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.run.seed)
            torch.cuda.reset_peak_memory_stats()

        cache_dir = ctx.cache_dir / adapter_name
        train_rows = load_parquet_split(cache_dir / "train.parquet", max_seq_length=eff.train.max_seq_length)
        val_rows = load_parquet_split(cache_dir / "val.parquet", max_seq_length=eff.train.max_seq_length)
        if not train_rows:
            raise ValueError(
                f"No training examples available for adapter {adapter_name!r} at max_seq_length={eff.train.max_seq_length}"
            )

        train_ds = hf_datasets.Dataset.from_list(train_rows)
        val_ds = hf_datasets.Dataset.from_list(val_rows) if val_rows else None

        tiny_dir = ctx.cache_dir / "_tiny_gemma4"
        tokenizer, _, _, _ = load_served_tokenizer_and_template(cfg, tiny_fallback_dir=tiny_dir)

        base_model = load_base_model_for_training(cfg, eff, tiny_fallback_dir=tiny_dir)
        base_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id

        peft_model, target_regex, matched_modules = attach_lora_adapter(base_model, eff.lora)

        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
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

        # Resume checkpoint restoration (§7.8)
        resume_ckpt = resolve_resume_checkpoint(ctx, cfg, adapter_name)
        resumed_step = 0
        if resume_ckpt is not None:
            resume_meta = json.loads((resume_ckpt / "trainer_resume.json").read_text(encoding="utf-8"))
            resumed_step = int(resume_meta.get("step", 0))
            adapter_st = load_safetensors(str(resume_ckpt / "adapter" / "adapter_model.safetensors"), device="cpu")
            set_peft_model_state_dict(peft_model, adapter_st)
            if resume_meta.get("sampler_state"):
                sampler.load_state_dict(resume_meta["sampler_state"])
            if resume_meta.get("schedule_state"):
                wsd.load_resume_state(
                    resume_meta["schedule_state"],
                    new_t_start=t_start,
                    new_deadline=soft_deadline,
                    extend=cfg.resume.extend,
                )
            _restore_rng_states(resume_ckpt / "rng_state.pt")
            notes.append(f"Resumed from {resume_ckpt} at step {resumed_step}")
            logger.info("[%s] Restored checkpoint %s (step=%d)", adapter_name, resume_ckpt, resumed_step)

        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=eff.train.lr,
            weight_decay=eff.train.weight_decay,
        )
        if resume_ckpt is not None and (resume_ckpt / "optimizer.pt").is_file():
            try:
                opt_state = torch.load(resume_ckpt / "optimizer.pt", map_location="cpu", weights_only=False)
                optimizer.load_state_dict(opt_state)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore optimizer state (%s); continuing with fresh moments", exc)

        scheduler = create_wsd_scheduler(optimizer, wsd)
        if resume_ckpt is not None and (resume_ckpt / "scheduler.pt").is_file():
            try:
                sched_state = torch.load(resume_ckpt / "scheduler.pt", map_location="cpu", weights_only=False)
                scheduler.load_state_dict(sched_state)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not restore scheduler state: %s", exc)

        has_cuda = torch.cuda.is_available()
        use_bf16 = eff.model_quantization in {"bf16", "nf4"} and has_cuda
        remaining_steps = max(1, total_epoch_steps - resumed_step)

        ckpt_root = ctx.checkpoints_dir / adapter_name
        ckpt_root.mkdir(parents=True, exist_ok=True)

        sft_args = SFTConfig(
            output_dir=str(ckpt_root / "hf_trainer"),
            num_train_epochs=float(eff.train.max_epochs),
            max_steps=remaining_steps,
            per_device_train_batch_size=eff.train.per_device_batch,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=eff.train.grad_accum,
            learning_rate=eff.train.lr,
            weight_decay=eff.train.weight_decay,
            max_grad_norm=eff.train.max_grad_norm,
            bf16=use_bf16,
            fp16=False,
            logging_steps=eff.train.log_every_steps,
            logging_first_step=True,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=cfg.run.seed,
            dataloader_num_workers=0,
            remove_unused_columns=True,
            average_tokens_across_devices=True,
            ddp_find_unused_parameters=not cfg.model.skip_multimodal_towers,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=None,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
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
        )

        collator = WeightedLanguageModelingCollator(pad_token_id=tokenizer.pad_token_id)
        trainer_cls = build_selective_sft_trainer_cls()
        trainer = trainer_cls(
            model=peft_model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
            callbacks=[cb],
            chunk_rows=eff.loss.chunk_rows,
            loss_impl=eff.loss.impl,
            custom_train_sampler=sampler,
            metrics_tracker=metrics_tracker,
        )
        cb.bind_trainer(trainer)

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
        train_result = trainer.train()
        cb.finalize_checks(trainer.model)

        final_val_loss: float | None = None
        if val_ds is not None and (deadline - time.time()) > 5.0:
            eval_metrics = trainer.evaluate()
            if "eval_loss" in eval_metrics:
                final_val_loss = round(float(eval_metrics["eval_loss"]), 6)

        final_adapter_dir = ckpt_root / "final_adapter"
        parity_path: Path | None = None
        if is_main:
            unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            final_adapter_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(final_adapter_dir)
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

            if cfg.parity.enabled and (deadline - time.time()) > 3.0:
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

        steps_completed = resumed_step + int(trainer.state.global_step)
        schedule_note = None
        if not wsd.ever_decayed and eff.schedule.decay_frac > 0.0 and steps_completed > 1:
            schedule_note = "Training ended before entering WSD decay phase"
            notes.append(schedule_note)

        summary = {
            "adapter": adapter_name,
            "steps_completed": steps_completed,
            "segment_steps": int(trainer.state.global_step),
            "epochs_completed": round(float(trainer.state.epoch or 0.0), 4),
            "final_train_loss": round(float(train_result.training_loss), 6),
            "final_val_loss": final_val_loss,
            "decay_entered": wsd.ever_decayed,
            "final_lr_multiplier": round(float(wsd.last_multiplier), 6),
            "dead_lora_check": cb.dead_check_result,
            "throughput_projection": cb.throughput_projection,
            "matched_module_count": len(matched_modules),
            "final_adapter_dir": str(final_adapter_dir),
            "parity_hf_path": str(parity_path) if parity_path else None,
            "oom_override": oom_override,
        }

        if is_main:
            t_end = time.time()
            manifest = ctx.read_manifest()
            tr_map = dict(manifest.get("training_summaries", {}))
            tr_map[adapter_name] = summary
            ctx.update_manifest({"training_summaries": tr_map})

            status = PhaseStatus(
                phase=phase_name,
                state="ok",
                started=utc_iso(t_start),
                ended=utc_iso(t_end),
                started_ts=t_start,
                ended_ts=t_end,
                planned_deadline=utc_iso(deadline),
                planned_deadline_ts=deadline,
                planned_duration_sec=round(deadline - t_start, 2),
                actual_duration_sec=round(t_end - t_start, 2),
                outputs=summary,
                notes=notes,
            )
            ctx.write_status(status)
            logger.info(
                "[%s] Finished %d step(s) in %.2fs (train_loss=%.4f, val_loss=%s)",
                adapter_name,
                steps_completed,
                t_end - t_start,
                summary["final_train_loss"],
                final_val_loss,
            )
            result_dict = status.to_dict()
        else:
            result_dict = {"phase": phase_name, "state": "ok", "outputs": summary}

        del trainer, peft_model, base_model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result_dict

    except Exception as exc:
        if is_main:
            t_end = time.time()
            logger.error("[%s] Training failed: %s", adapter_name, exc)
            status = PhaseStatus(
                phase=phase_name,
                state="failed",
                started=utc_iso(t_start),
                ended=utc_iso(t_end),
                started_ts=t_start,
                ended_ts=t_end,
                planned_deadline=utc_iso(deadline),
                planned_deadline_ts=deadline,
                planned_duration_sec=round(deadline - t_start, 2),
                actual_duration_sec=round(t_end - t_start, 2),
                error=str(exc),
                notes=notes,
            )
            ctx.write_status(status)
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
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
