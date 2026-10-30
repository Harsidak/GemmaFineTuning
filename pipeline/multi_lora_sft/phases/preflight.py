"""Phase 2 (`preflight`): data contract, leakage gate, chat template parity, and Parquet caching.

Implements design_doc.md §6.2, §7.1, §7.2, and §11:
- Hard-fails the whole preflight on any leakage against `tasks.jsonl` or broken global inputs.
- Isolates per-adapter dataset errors so if one adapter fails role/data filtering, remaining
  valid adapters still proceed to training (§6.2).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..config import PipelineConfig, load_config
from ..runctx import PhaseStatus, RunContext, configure_phase_logging, utc_iso
from ..training.data import (
    DataContractError,
    check_leakage,
    check_tasks_file_optional,
    load_raw_trajectories_checked,
    load_served_tokenizer_and_template,
    prepare_adapter_dataset,
)


def run_preflight_phase(
    ctx: RunContext,
    cfg: PipelineConfig | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Executes preflight validation and per-adapter Parquet tokenization/caching."""
    logger = configure_phase_logging(ctx.log_path("preflight"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)

    notes: list[str] = []
    report: dict[str, Any] = {
        "train_files": cfg.data.train_files,
        "tasks_file": cfg.data.tasks_file,
        "adapters": {},
        "valid_adapters": [],
        "failed_adapters": {},
    }

    try:
        raw_records, data_file_hashes = load_raw_trajectories_checked(cfg.data.train_files)
        report["raw_trajectory_count"] = len(raw_records)
        report["data_file_hashes"] = data_file_hashes

        # Leakage check (§7.1, §11)
        if cfg.eval.enabled and cfg.data.tasks_file and not Path(cfg.data.tasks_file).is_file():
            raise DataContractError(
                f"eval.enabled=True, but data.tasks_file does not exist: {cfg.data.tasks_file}"
            )
        test_tasks = check_tasks_file_optional(cfg.data.tasks_file)
        report["test_tasks_checked"] = len(test_tasks)

        leakage = check_leakage(
            raw_records,
            test_tasks,
            eval_task_ids=(cfg.eval.task_ids if cfg.eval.enabled else None),
        )
        report["leakage"] = leakage.to_dict()
        if leakage.same_repo_warnings:
            msg = f"{len(leakage.same_repo_warnings)} training trajectory(ies) share repo with held-out tasks (different commit)"
            notes.append(msg)
            logger.warning(msg)

        if not leakage.passed:
            ctx.preflight_report_path.parent.mkdir(parents=True, exist_ok=True)
            ctx.preflight_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise DataContractError(f"Benchmark leakage gate FAILED: {leakage.to_dict()}")

        # Load served tokenizer and chat template (§7.1, §7.2)
        tiny_dir = ctx.cache_dir / "_tiny_gemma4"
        tokenizer, chat_template, tpl_sha, tok_sha = load_served_tokenizer_and_template(
            cfg, tiny_fallback_dir=tiny_dir
        )
        served_tok_path = cfg.model.served_path or cfg.model.train_base_path
        if served_tok_path == "__tiny__":
            served_tok_path = str(tiny_dir)

        report["template_sha256"] = tpl_sha
        report["tokenizer_sha256"] = tok_sha

        # Per-adapter dataset tokenization & Parquet cache (§7.2, §6.2)
        supervised_tokens_by_adapter: dict[str, int] = {}
        for spec in cfg.adapters:
            eff = cfg.resolve_adapter(spec)
            adapter_cache_dir = ctx.cache_dir / spec.name
            try:
                summary = prepare_adapter_dataset(
                    cfg=cfg,
                    eff_adapter=eff,
                    raw_records=raw_records,
                    data_file_hashes=data_file_hashes,
                    tokenizer=tokenizer,
                    served_tokenizer_path=served_tok_path,
                    chat_template=chat_template,
                    template_sha256=tpl_sha,
                    tokenizer_sha256=tok_sha,
                    out_cache_dir=adapter_cache_dir,
                )
                report["adapters"][spec.name] = summary
                report["valid_adapters"].append(spec.name)
                supervised_tokens_by_adapter[spec.name] = int(summary["train_supervised_tokens"])
                logger.info(
                    "Adapter %r preflight OK: %d train (%d sup tokens), %d val, overlong=%d, dup=%d",
                    spec.name,
                    summary["train_examples"],
                    summary["train_supervised_tokens"],
                    summary["val_examples"],
                    summary["dropped_overlong"],
                    summary["dropped_duplicates"],
                )
            except Exception as adapter_exc:  # noqa: BLE001
                err_str = str(adapter_exc)
                report["failed_adapters"][spec.name] = err_str
                notes.append(f"Adapter {spec.name!r} failed preflight and will be skipped: {err_str}")
                logger.error("Adapter %r failed preflight: %s", spec.name, err_str)

        ctx.preflight_report_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.preflight_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        ctx.update_manifest(
            {
                "template_sha256": tpl_sha,
                "tokenizer_sha256": tok_sha,
                "data_file_hashes": data_file_hashes,
                "leakage_passed": True,
                "preflight_valid_adapters": report["valid_adapters"],
                "preflight_failed_adapters": report["failed_adapters"],
                "supervised_tokens_by_adapter": supervised_tokens_by_adapter,
            }
        )

        if not report["valid_adapters"]:
            raise DataContractError(
                f"All configured adapters failed preflight: {report['failed_adapters']}"
            )

        t_end = time.time()
        status = PhaseStatus(
            phase="preflight",
            state="ok",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            outputs={
                "report_path": str(ctx.preflight_report_path),
                "valid_adapters": report["valid_adapters"],
                "failed_adapters": report["failed_adapters"],
                "supervised_tokens_by_adapter": supervised_tokens_by_adapter,
            },
            notes=notes,
        )
        ctx.write_status(status)
        return status.to_dict()

    except Exception as exc:
        t_end = time.time()
        logger.error("Preflight failed: %s", exc)
        ctx.update_manifest({"leakage_passed": report.get("leakage", {}).get("passed", False)})
        status = PhaseStatus(
            phase="preflight",
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
            outputs={"report_path": str(ctx.preflight_report_path)},
            notes=notes,
        )
        ctx.write_status(status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 2: preflight")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    try:
        run_preflight_phase(ctx, deadline=args.deadline)
        return 0
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
