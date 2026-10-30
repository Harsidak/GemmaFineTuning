"""Phase 1 (`setup`): offline wheel installation, import checks, GPU & input path verification.

Implements design_doc.md §5, §6.2, and §10.2.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..config import PipelineConfig, load_config
from ..runctx import (
    PhaseStatus,
    RunContext,
    collect_package_versions,
    configure_phase_logging,
    utc_iso,
)

REQUIRED_CORE_IMPORTS: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "safetensors",
    "pyarrow",
    "yaml",
)

TUNIX_CORE_IMPORTS: tuple[str, ...] = (
    "transformers",
    "safetensors",
    "pyarrow",
    "yaml",
    "jax",
    "flax",
    "optax",
    "qwix",
    "tunix",
)

OPTIONAL_EVAL_IMPORTS: tuple[str, ...] = (
    "vllm",
    "swegemma",
    "adk_submission",
)


def install_wheels_from_dirs(wheelhouse_dirs: list[str], staging_dir: Path, logger: Any) -> list[str]:
    """Installs offline wheels from `wheelhouse_dirs`, restoring stripped `+cu128` suffixes if needed."""
    installed_sources: list[str] = []
    staging_dir.mkdir(parents=True, exist_ok=True)

    for wh_str in wheelhouse_dirs:
        wh = Path(wh_str)
        if not wh.is_dir():
            raise FileNotFoundError(f"Configured wheelhouse directory does not exist: {wh}")
        wheels = sorted(wh.glob("*.whl"))
        if not wheels:
            logger.warning("Wheelhouse directory %s contains 0 .whl files; skipping", wh)
            continue

        # Stage wheels with `+cu128` restored if Kaggle stripped `+` from torch filenames (§10.2)
        staged_wh = staging_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", wh.name)
        staged_wh.mkdir(parents=True, exist_ok=True)
        staged_files: list[str] = []
        for w in wheels:
            target_name = w.name
            if target_name.startswith("torch-") and "cu128" in target_name and "+cu128" not in target_name:
                target_name = target_name.replace("cu128", "+cu128", 1)
            dst = staged_wh / target_name
            if not dst.exists():
                shutil.copy2(w, dst)
            staged_files.append(str(dst))

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(staged_wh),
            *staged_files,
        ]
        logger.info("Installing %d wheel(s) from %s", len(staged_files), wh)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Offline wheel install failed for {wh} (rc={proc.returncode}): {proc.stderr[-1000:]}"
            )
        installed_sources.append(str(wh))

    return installed_sources


def run_setup_phase(ctx: RunContext, cfg: PipelineConfig | None = None, deadline: float | None = None) -> dict[str, Any]:
    """Executes the `setup` phase and updates `status/setup.json` and `manifest.json`."""
    logger = configure_phase_logging(ctx.log_path("setup"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)

    notes: list[str] = []
    outputs: dict[str, Any] = {}

    try:
        if cfg.setup.install_wheels and cfg.setup.wheelhouse_dirs:
            installed_from = install_wheels_from_dirs(
                cfg.setup.wheelhouse_dirs, ctx.run_dir / "wheel_staging", logger
            )
            outputs["installed_wheelhouses"] = installed_from

        from ..shared.backend import resolve_backend

        backends = {
            spec.name: resolve_backend(
                cfg.resolve_adapter(spec).train.backend,
                served_model_id=cfg.effective_served_model_id(),
                accelerator=cfg.resolve_adapter(spec).train.accelerator,
            )
            for spec in cfg.adapters
        }
        outputs["backends"] = backends
        uses_tunix = "tunix" in backends.values()
        uses_trl = "trl" in backends.values()
        required = tuple(dict.fromkeys([*(REQUIRED_CORE_IMPORTS if uses_trl else ()), *(TUNIX_CORE_IMPORTS if uses_tunix else ())]))

        missing_core: list[str] = []
        for mod_name in required:
            try:
                importlib.import_module(mod_name)
            except Exception as exc:  # noqa: BLE001
                missing_core.append(f"{mod_name} ({exc})")
        if missing_core:
            raise RuntimeError(f"Missing required core imports: {', '.join(missing_core)}")

        eval_import_status: dict[str, bool] = {}
        for mod_name in OPTIONAL_EVAL_IMPORTS:
            try:
                importlib.import_module(mod_name)
                eval_import_status[mod_name] = True
            except Exception:  # noqa: BLE001
                eval_import_status[mod_name] = False
        outputs["eval_imports"] = eval_import_status
        if cfg.eval.enabled and not all(eval_import_status.values()):
            missing_eval = [k for k, v in eval_import_status.items() if not v]
            notes.append(f"Eval enabled, but optional eval packages missing at setup: {missing_eval}")
            logger.warning("Optional eval packages missing: %s", missing_eval)

        from ..training.device import accelerator_inventory, resolve_accelerator

        inventory = accelerator_inventory()
        gpus: list[dict[str, Any]] = inventory["cuda"]
        gpu_count = len(gpus)
        accelerators: dict[str, str] = {}
        for spec in cfg.adapters:
            requested = cfg.resolve_adapter(spec).train.accelerator
            if backends[spec.name] == "tunix":
                from tpu_training.runtime import resolve_platform

                accelerators[spec.name] = resolve_platform(requested)  # JAX view: tpu | cpu (raises if a TPU was demanded)
            else:
                accelerators[spec.name] = resolve_accelerator(requested)  # raises with an install hint if missing
        outputs["accelerator_inventory"] = inventory
        outputs["resolved_accelerators"] = accelerators
        uses_tpu = "tpu" in accelerators.values()

        if cfg.setup.require_gpu and gpu_count == 0 and not uses_tpu:
            raise RuntimeError("setup.require_gpu=True, but torch.cuda.is_available() is False and no TPU is in use")
        if cfg.setup.expected_gpu_count is not None and not uses_tpu and gpu_count != cfg.setup.expected_gpu_count:
            raise RuntimeError(
                f"Expected {cfg.setup.expected_gpu_count} GPU(s), found {gpu_count}"
            )
        if "cpu" in accelerators.values() and cfg.run.mode == "full" and cfg.model.train_base_path not in {None, "__tiny__"}:
            msg = "Full-mode training resolved to CPU (no GPU/TPU visible); this will be extremely slow"
            notes.append(msg)
            logger.warning(msg)
        if uses_tpu and cfg.eval.enabled:
            msg = "Eval uses vLLM on CUDA GPUs; on a TPU session set eval.enabled=false (the zip is still packaged)"
            notes.append(msg)
            logger.warning(msg)

        # Input path verification (§6.2)
        if not cfg.data.train_files:
            raise FileNotFoundError("data.train_files is empty; at least one training JSONL is required")
        for tf in cfg.data.train_files:
            if not Path(tf).is_file():
                raise FileNotFoundError(f"Training file not found: {tf}")

        from ..training.model import check_train_served_compat, validate_model_dir

        if cfg.model.train_base_path is None and cfg.model.served_path is None:
            raise FileNotFoundError("model.train_base_path is not configured")

        train_hf_cfg = served_hf_cfg = None
        if cfg.model.train_base_path and cfg.model.train_base_path != "__tiny__":
            train_hf_cfg = validate_model_dir(cfg.model.train_base_path, "model.train_base_path", for_training=True)
        elif cfg.model.train_base_path is None:
            notes.append("model.train_base_path is unset; training will fail unless every adapter is resumed/fallback")
        if cfg.model.served_path and cfg.model.served_path != "__tiny__":
            served_hf_cfg = validate_model_dir(cfg.model.served_path, "model.served_path", for_training=False)
        if train_hf_cfg is not None and served_hf_cfg is not None:
            check_train_served_compat(train_hf_cfg, served_hf_cfg)

        out_root = Path(cfg.run.output_root)
        probe = out_root if out_root.exists() else next((p for p in out_root.parents if p.exists()), None)
        if probe is not None:
            free_gib = shutil.disk_usage(probe).free / 1024**3
            outputs["output_root_free_gib"] = round(free_gib, 2)
            min_needed = max(cfg.resolve_adapter(a).train.min_free_disk_gib for a in cfg.adapters) + 1.0
            if free_gib < min_needed:
                msg = f"Only {free_gib:.1f} GiB free under {probe}; checkpoints may be skipped (need > {min_needed:.1f})"
                notes.append(msg)
                logger.warning(msg)

        if cfg.submission.template_dir and not Path(cfg.submission.template_dir).is_dir():
            raise FileNotFoundError(f"submission.template_dir does not exist: {cfg.submission.template_dir}")

        pkg_versions = collect_package_versions()
        outputs["gpu_count"] = gpu_count
        outputs["gpus"] = gpus
        outputs["python_version"] = platform.python_version()

        ctx.update_manifest(
            {
                "python_version": platform.python_version(),
                "gpus": gpus,
                "tpu": inventory.get("tpu"),
                "resolved_accelerators": accelerators,
                "package_versions": pkg_versions,
            }
        )

        t_end = time.time()
        status = PhaseStatus(
            phase="setup",
            state="ok",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            outputs=outputs,
            notes=notes,
        )
        ctx.write_status(status)
        logger.info("Setup complete in %.2fs (GPUs=%d)", t_end - t_start, gpu_count)
        return status.to_dict()

    except Exception as exc:
        t_end = time.time()
        logger.error("Setup failed: %s", exc)
        status = PhaseStatus(
            phase="setup",
            state="failed",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            error=str(exc),
            outputs=outputs,
            notes=notes,
        )
        ctx.write_status(status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 1: setup")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    try:
        run_setup_phase(ctx, deadline=args.deadline)
        return 0
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
