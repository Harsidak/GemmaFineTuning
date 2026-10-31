"""Optuna entry point (`python -m pipeline.multi_lora_sft.tune`).

    python -m pipeline.multi_lora_sft.tune --config pipeline/multi_lora_sft/configs/e4b_tpu.yaml \\
        --set tuning.trials=8 --set tuning.trial_minutes=10 --set model.train_base_path=...

Backend-agnostic: every trial is a normal orchestrator run of the configured trainer (TRL or Tunix) with W&B off.
Outputs `<tuning storage>/best_params.json`, which the next normal run (or the notebook) applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .shared.bundle import discover_bundles, previous_session_count, sync_bundle
from .shared.tuning import TuningError, run_study


def _write_bundle(cfg, study_dir) -> None:
    """`tuning/{study.db,best_params.json,trials.json}` in the output bundle, so the next session continues the same study
    and a normal run can apply the best parameters. Never raises: the study results are already on disk."""
    out = cfg.output.bundle_dir
    if not out:
        return
    try:
        attached = [Path(p) for p in cfg.resume.bundle_dirs] + (discover_bundles([cfg.resume.input_root]) if cfg.resume.auto_discover else [])
        sync_bundle(
            out, run_checkpoints_dir=Path(study_dir), adapters=[], served_model_id=cfg.effective_served_model_id(),
            dataset_id=cfg.output.dataset_id, dataset_title=cfg.output.dataset_title, git_sha=cfg.run.git_sha,
            tuning_dir=study_dir, previous_sessions=previous_session_count(attached), scan_secrets=cfg.output.scan_secrets,
            extra={"kind": "tuning"},
        )
        print(f"tuning bundle written to {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not write the tuning bundle: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optuna hyper-parameter search for the multi-LoRA SFT pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between trial progress checks")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config, args.sets)  # fail fast on a bad config, before any trial starts
        result = run_study(args.config, args.sets, poll_sec=args.poll)
        _write_bundle(cfg, result.study_dir)
    except (ConfigError, TuningError) as exc:
        print(f"tuning failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.best_params is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
