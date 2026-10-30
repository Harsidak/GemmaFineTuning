"""Phase 4 (`export`): CPU bf16 cast, strict safetensors key & sanity checks, and `adapter_config.json` rewrite.

Implements design_doc.md §6.2 and §8.1.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ..config import (
    ADAPTER_NAME_RE,
    MAX_LORA_RANK,
    EffectiveAdapterConfig,
    PipelineConfig,
    load_config,
)
from ..runctx import (
    PhaseStatus,
    RunContext,
    configure_phase_logging,
    sha256_file,
    utc_iso,
)
from ..training.callbacks import find_latest_checkpoint

EXPORT_KEY_RE = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$"
)


class ExportValidationError(ValueError):
    """Raised when an adapter checkpoint violates export key, dtype, rank, or numerical sanity rules."""


def format_served_base_model_id(served_model_id: str) -> str:
    """Formats `base_model_name_or_path` as `google/<served_model_id>` matching sample submission."""
    s = served_model_id.strip()
    if "/" in s:
        return s
    return f"google/{s}"


def normalize_peft_key(key: str) -> str:
    """Normalizes `.lora_A.default.weight` -> `.lora_A.weight` if raw state_dict keys are encountered."""
    return re.sub(r"\.lora_([AB])\.default\.weight$", r".lora_\1.weight", key)


def export_single_adapter(
    *,
    adapter_name: str,
    source_adapter_dir: str | Path,
    output_adapter_dir: str | Path,
    served_model_id: str,
    eff_adapter: EffectiveAdapterConfig | None = None,
    allow_all_zero_b: bool = False,
) -> dict[str, Any]:
    """Exports and validates a single PEFT LoRA adapter directory on CPU in `bfloat16` (§8.1)."""
    import torch
    from safetensors.torch import load_file, save_file

    if not ADAPTER_NAME_RE.match(adapter_name):
        raise ExportValidationError(f"Invalid adapter name {adapter_name!r}")

    src_dir = Path(source_adapter_dir)
    st_path = src_dir / "adapter_model.safetensors"
    cfg_path = src_dir / "adapter_config.json"
    if not st_path.is_file():
        raise ExportValidationError(f"Missing adapter_model.safetensors in {src_dir}")
    if not cfg_path.is_file():
        raise ExportValidationError(f"Missing adapter_config.json in {src_dir}")

    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_tensors = load_file(str(st_path), device="cpu")
    if not raw_tensors:
        raise ExportValidationError(f"adapter_model.safetensors in {src_dir} contains 0 tensors")

    cast_tensors: dict[str, torch.Tensor] = {}
    modules_with_a: dict[str, tuple[int, int]] = {}
    modules_with_b: dict[str, tuple[int, int]] = {}
    non_zero_b_count = 0

    for orig_key, tensor in raw_tensors.items():
        key = normalize_peft_key(orig_key)
        if any(bad in key for bad in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
            raise ExportValidationError(
                f"Forbidden multimodal key in adapter {adapter_name!r}: {key!r}"
            )
        m = EXPORT_KEY_RE.match(key)
        if not m:
            raise ExportValidationError(
                f"Adapter {adapter_name!r} key {key!r} violates required format {EXPORT_KEY_RE.pattern}"
            )

        if tensor.ndim != 2:
            raise ExportValidationError(f"Expected 2D LoRA weight for {key!r}, got shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ExportValidationError(f"NaN or Inf detected in adapter {adapter_name!r} tensor {key!r}")

        bf16_t = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if not torch.isfinite(bf16_t).all():
            raise ExportValidationError(f"Overflow when casting {key!r} to bfloat16")

        cast_tensors[key] = bf16_t
        # Module path relative to `Gemma4ForConditionalGeneration` (`base_model.model.`)
        mod_rel = f"model.language_model.layers.{m.group(1)}.{m.group(2)}"
        ab = m.group(3)
        if ab == "A":
            modules_with_a[mod_rel] = (int(bf16_t.shape[0]), int(bf16_t.shape[1]))
        else:
            modules_with_b[mod_rel] = (int(bf16_t.shape[0]), int(bf16_t.shape[1]))
            if float(bf16_t.float().norm().item()) > 0.0:
                non_zero_b_count += 1

    if set(modules_with_a.keys()) != set(modules_with_b.keys()):
        missing_b = sorted(set(modules_with_a) - set(modules_with_b))
        missing_a = sorted(set(modules_with_b) - set(modules_with_a))
        raise ExportValidationError(
            f"Mismatched lora_A / lora_B keys in {adapter_name!r}: missing_B={missing_b[:3]}, missing_A={missing_a[:3]}"
        )

    detected_ranks = {shape[0] for shape in modules_with_a.values()} | {
        shape[1] for shape in modules_with_b.values()
    }
    if len(detected_ranks) != 1:
        raise ExportValidationError(
            f"Inconsistent LoRA ranks across modules in {adapter_name!r}: {sorted(detected_ranks)}"
        )
    rank = next(iter(detected_ranks))
    expected_r = eff_adapter.lora.r if eff_adapter is not None else int(raw_cfg.get("r", rank))
    if rank != expected_r:
        raise ExportValidationError(
            f"Adapter {adapter_name!r} tensor rank {rank} != configured r={expected_r}"
        )
    if not (1 <= rank <= MAX_LORA_RANK):
        raise ExportValidationError(f"Adapter {adapter_name!r} rank {rank} exceeds {MAX_LORA_RANK}")

    if not allow_all_zero_b and non_zero_b_count == 0:
        raise ExportValidationError(
            f"All {len(modules_with_b)} lora_B matrices in {adapter_name!r} are identically zero (untrained adapter)"
        )

    matched_meta_path = src_dir / "matched_target_modules.json"
    sorted_modules = sorted(modules_with_a.keys())
    if matched_meta_path.is_file():
        matched_meta = json.loads(matched_meta_path.read_text(encoding="utf-8"))
        expected_mods = sorted(matched_meta.get("matched_modules", []))
        if expected_mods and expected_mods != sorted_modules:
            raise ExportValidationError(
                f"Adapter {adapter_name!r} weight modules do not match target_modules ({len(sorted_modules)} vs {len(expected_mods)})"
            )

    export_mode = eff_adapter.lora.export_target_modules_mode if eff_adapter is not None else "explicit"
    if export_mode == "explicit":
        target_modules_out: list[str] | str = sorted_modules
    else:
        target_modules_out = (
            raw_cfg.get("target_modules")
            if isinstance(raw_cfg.get("target_modules"), str)
            else (eff_adapter.lora.target_regex if eff_adapter is not None else sorted_modules)
        )

    alpha = int(eff_adapter.lora.alpha) if eff_adapter is not None else int(raw_cfg.get("lora_alpha", rank * 2))
    out_cfg: dict[str, Any] = dict(raw_cfg)
    out_cfg.update(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": format_served_base_model_id(served_model_id),
            "inference_mode": True,
            "r": int(rank),
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "bias": eff_adapter.lora.bias if eff_adapter is not None else str(raw_cfg.get("bias", "none")),
            "use_rslora": bool(eff_adapter.lora.use_rslora) if eff_adapter is not None else bool(raw_cfg.get("use_rslora", False)),
            "target_modules": target_modules_out,
        }
    )

    out_dir = Path(output_adapter_dir)
    tmp_dir = out_dir.parent / f".tmp_export_{adapter_name}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_st = tmp_dir / "adapter_model.safetensors"
    out_json = tmp_dir / "adapter_config.json"
    save_file(cast_tensors, str(out_st))
    out_json.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    st_bytes = out_st.stat().st_size
    if st_bytes < 128 or st_bytes >= 3 * 1024 * 1024 * 1024:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ExportValidationError(f"Unreasonable exported safetensors size for {adapter_name!r}: {st_bytes} bytes")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(out_dir)

    return {
        "adapter": adapter_name,
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "r": int(rank),
        "lora_alpha": alpha,
        "module_count": len(sorted_modules),
        "non_zero_b_modules": non_zero_b_count,
        "safetensors_bytes": (out_dir / "adapter_model.safetensors").stat().st_size,
        "safetensors_sha256": sha256_file(out_dir / "adapter_model.safetensors"),
        "base_model_name_or_path": out_cfg["base_model_name_or_path"],
    }


def locate_trained_adapter_source(ctx: RunContext, adapter_name: str) -> tuple[Path | None, str | None]:
    """Finds `final_adapter` or latest wall-clock checkpoint `step_<n>/adapter` (§6.2)."""
    final_dir = ctx.checkpoints_dir / adapter_name / "final_adapter"
    if (final_dir / "adapter_model.safetensors").is_file():
        return final_dir, "final_adapter"
    latest_ckpt = find_latest_checkpoint(ctx.checkpoints_dir / adapter_name)
    if latest_ckpt is not None and (latest_ckpt / "adapter" / "adapter_model.safetensors").is_file():
        return latest_ckpt / "adapter", f"checkpoint:{latest_ckpt.name}"
    return None, None


def run_export_phase(
    ctx: RunContext,
    cfg: PipelineConfig | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Exports all available trained adapters to `runs/<run_id>/adapters/<name>/`."""
    logger = configure_phase_logging(ctx.log_path("export"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)
    served_id = cfg.effective_served_model_id()

    exported: dict[str, Any] = {}
    failed: dict[str, str] = {}
    notes: list[str] = []

    for spec in cfg.adapters:
        eff = cfg.resolve_adapter(spec)
        src_dir, src_kind = locate_trained_adapter_source(ctx, spec.name)
        if src_dir is None:
            msg = f"No trained final_adapter or checkpoint found for {spec.name!r}"
            failed[spec.name] = msg
            notes.append(msg)
            logger.warning(msg)
            continue

        try:
            info = export_single_adapter(
                adapter_name=spec.name,
                source_adapter_dir=src_dir,
                output_adapter_dir=ctx.adapters_dir / spec.name,
                served_model_id=served_id,
                eff_adapter=eff,
            )
            info["source_kind"] = src_kind
            exported[spec.name] = info
            if src_kind != "final_adapter":
                notes.append(f"Adapter {spec.name!r} exported from fallback {src_kind}")
            logger.info(
                "Exported adapter %r (%s): r=%d, modules=%d, size=%.2f MiB",
                spec.name,
                src_kind,
                info["r"],
                info["module_count"],
                info["safetensors_bytes"] / (1024 * 1024),
            )
        except Exception as exc:  # noqa: BLE001
            failed[spec.name] = str(exc)
            notes.append(f"Export failed for {spec.name!r}: {exc}")
            logger.error("Export failed for %r: %s", spec.name, exc)

    ctx.update_manifest({"exported_adapters": exported, "export_failures": failed})

    t_end = time.time()
    state = "ok" if exported else "failed"
    err = None if exported else f"0 adapters exported: {failed}"
    status = PhaseStatus(
        phase="export",
        state=state,
        started=utc_iso(t_start),
        ended=utc_iso(t_end),
        started_ts=t_start,
        ended_ts=t_end,
        planned_deadline=utc_iso(deadline) if deadline else None,
        planned_deadline_ts=deadline,
        planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
        actual_duration_sec=round(t_end - t_start, 2),
        error=err,
        outputs={"exported": exported, "failed": failed},
        notes=notes,
    )
    ctx.write_status(status)
    if state != "ok":
        raise ExportValidationError(err or "Export failed")
    return status.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 4: export")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    try:
        run_export_phase(ctx, deadline=args.deadline)
        return 0
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
