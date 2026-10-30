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

from .config import ConfigError, load_config
from .shared.tuning import TuningError, run_study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optuna hyper-parameter search for the multi-LoRA SFT pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between trial progress checks")
    args = ap.parse_args(argv)
    try:
        load_config(args.config, args.sets)  # fail fast on a bad config, before any trial starts
        result = run_study(args.config, args.sets, poll_sec=args.poll)
    except (ConfigError, TuningError) as exc:
        print(f"tuning failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.best_params is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
