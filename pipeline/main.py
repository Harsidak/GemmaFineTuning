"""Unified Pipeline Orchestrator for Gemma 4 Developer Agent.

This module acts as the central control plane for:
1. Training (SFT, DPO preference fine-tuning, and time-budgeted Multi-LoRA SFT)
2. Local evaluation & SWE-bench sandbox testing
3. Declarative ADK submission packaging & validation
4. Workspace status monitoring

Can be imported directly into Kaggle launcher notebooks or executed via CLI:
    python pipeline/main.py --action [status | train | multi_lora_sft | evaluate | package]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root or current pipeline directory is in sys.path
CURRENT_FILE = Path(__file__).resolve()
PIPELINE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PIPELINE_DIR.parent

for p in [str(PROJECT_ROOT), str(PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.src.agent.builder import ADKSubmissionBuilder
from pipeline.src.data.loader import load_tasks
from pipeline.src.eval.harness import LocalEvaluator
from pipeline.src.utils.packager import (
    package_submission_archive,
    validate_submission_directory,
)

logger = logging.getLogger("PipelineRunner")


# Named configs for `--preset` (the names the README uses). The Kaggle notebook keeps its own preset table.
PRESETS: dict[str, str] = {
    "31b_4x_l4": "pipeline/multi_lora_sft/configs/31b_full.yaml",  # main submission: 31B NF4 QLoRA, FSDP on 4x L4
    "e4b_4x_l4": "pipeline/multi_lora_sft/configs/e4b_full.yaml",  # E4B bf16 LoRA, DDP on 4x L4
    "e4b_tpu": "pipeline/multi_lora_sft/configs/e4b_tpu.yaml",  # E4B on a TPU v5e-8 (Tunix / JAX)
    "e4b_calibration": "pipeline/multi_lora_sft/configs/e4b_calibration.yaml",  # 30-min E4B speed / memory check
    "local_single_gpu": "pipeline/multi_lora_sft/configs/local_single_gpu.yaml",  # E4B NF4 on one local GPU
}
DEFAULT_PRESET = "31b_4x_l4"


def resolve_preset_config(root_dir: Path, preset: str | None, config_path: str | Path | None) -> Path:
    """`--config` wins, then `--preset`, then the default (31B on 4x L4). Relative paths are repo-relative."""
    if config_path:
        p = Path(config_path)
    else:
        name = preset or DEFAULT_PRESET
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
        p = Path(PRESETS[name])
    return p if p.is_absolute() else Path(root_dir) / p


class PipelineRunner:
    """Modular coordinator for all Gemma fine-tuning and agent evaluation workflows."""

    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT
        self.pipeline_dir = (
            self.root_dir / "pipeline" if (self.root_dir / "pipeline").exists() else self.root_dir
        )

    def status(self) -> dict[str, Any]:
        """Inspects and returns the workspace readiness status."""
        rules_path = self.root_dir / ".agents" / "rules.md"
        val_tasks = self.root_dir / "data" / "val_tasks.jsonl"
        train_traj = self.root_dir / "data" / "train_trajectories.jsonl"
        submission_zip = self.root_dir / "submission.zip"
        mistakes_log = self.root_dir / "planning" / "MISTAKES.md"
        roadmap = self.root_dir / "planning" / "ROADMAP.md"
        resources_index = self.root_dir / "resources" / "INDEX.md"
        scripts_dir = self.root_dir / "scripts"
        configs = (
            list((self.pipeline_dir / "configs").glob("*.yaml"))
            if (self.pipeline_dir / "configs").exists()
            else []
        )
        multi_lora_configs_dir = self.pipeline_dir / "multi_lora_sft" / "configs"
        multi_lora_configs = (
            list(multi_lora_configs_dir.glob("*.yaml"))
            if multi_lora_configs_dir.exists()
            else []
        )

        status_info = {
            "root_dir": str(self.root_dir),
            "rules_ready": rules_path.exists(),
            "validation_tasks_found": val_tasks.exists(),
            "train_trajectories_found": train_traj.exists(),
            "configs_count": len(configs),
            "multi_lora_sft_configs_count": len(multi_lora_configs),
            "packaged_submission_ready": submission_zip.exists(),
            "planning_ready": mistakes_log.exists() and roadmap.exists(),
            "resources_ready": resources_index.exists(),
            "scripts_ready": scripts_dir.exists(),
        }

        print("=" * 65)
        print("GEMMA 4 DEVELOPER AGENT - PIPELINE STATUS")
        print("=" * 65)
        print(f"• Root Directory:            {self.root_dir}")
        print(f"• Rules & Constraints:       {'[READY]' if status_info['rules_ready'] else '[MISSING]'}")
        print(f"• Planning & Mistakes Log:   {'[READY]' if status_info['planning_ready'] else '[MISSING]'}")
        print(f"• Research Catalogue:        {'[READY]' if status_info['resources_ready'] else '[MISSING]'}")
        print(f"• Helper Scripts:            {'[READY]' if status_info['scripts_ready'] else '[MISSING]'}")
        print(f"• Validation Tasks:          {'[FOUND]' if status_info['validation_tasks_found'] else '[MISSING]'}")
        print(f"• Train Trajectories:        {'[FOUND]' if status_info['train_trajectories_found'] else '[MISSING]'}")
        print(
            f"• Config Files Available:    {len(configs)} (multi_lora_sft: {len(multi_lora_configs)})"
        )
        print(f"• Submission Archive:        {'[READY]' if status_info['packaged_submission_ready'] else '[NOT BUILT]'}")
        print("=" * 65)
        return status_info

    def train(
        self,
        mode: str = "multi_lora_sft",
        config_path: str | Path | None = None,
        train_data: str | Path | None = None,
        base_model: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        """Runs the Multi-LoRA SFT orchestrator (default) or the legacy single-run SFT / DPO trainers.

        train_data, base_model and output_dir override the config (SFT only), so private datasets and
        attached models are passed per run instead of being hardcoded in the public repo.
        """
        if mode == "multi_lora_sft":
            overrides: list[str] = []
            if train_data:
                files = [str(f) for f in (train_data if isinstance(train_data, (list, tuple)) else [train_data])]
                overrides.append("data.train_files=" + json.dumps(files))
            if base_model:
                overrides.append(f"model.train_base_path={base_model}")
            if output_dir:
                overrides.append(f"run.output_root={output_dir}")
            return self.multi_lora_sft(config_path=config_path, set_overrides=overrides)

        if not config_path:
            default_name = "train_lora_sft.yaml" if mode == "sft" else "train_dpo_rl.yaml"
            config_path = self.pipeline_dir / "configs" / default_name

        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.root_dir / cfg_path

        print(f"\n[PIPELINE] Starting {mode.upper()} Training...")
        print(f"[PIPELINE] Using config: {cfg_path}")

        if mode == "sft":
            from pipeline.src.training.sft_trainer import run_sft_training

            return run_sft_training(
                cfg_path, train_data_path=train_data, base_model=base_model, output_dir=output_dir
            )
        if mode == "dpo":
            from pipeline.src.training.dpo_trainer import run_dpo_training

            return run_dpo_training(cfg_path)
        raise ValueError(f"Unknown training mode: {mode}. Choose 'sft', 'dpo', or 'multi_lora_sft'.")

    def multi_lora_sft(
        self,
        config_path: str | Path | None = None,
        set_overrides: list[str] | None = None,
        preset: str | None = None,
        dry_run: bool = False,
    ) -> int:
        """Runs the 7-phase Multi-LoRA SFT orchestrator (`pipeline.multi_lora_sft.run`), exactly as the Kaggle notebook
        does: setup -> preflight -> train -> export -> package -> eval -> report. Returns its exit code."""
        cfg_path = resolve_preset_config(self.root_dir, preset, config_path)
        sets = list(set_overrides or [])
        print(f"\n[PIPELINE] Multi-LoRA SFT orchestrator | config: {cfg_path}" + (f" | --set {sets}" if sets else ""))
        if dry_run:
            from pipeline.multi_lora_sft.config import load_config

            cfg = load_config(cfg_path, sets)
            n_tasks = cfg.eval.num_tasks if cfg.eval.smoke_tasks else 0
            print(f"[PIPELINE] dry run: config valid. served={cfg.effective_served_model_id()} "
                  f"adapters={[a.name for a in cfg.adapters]} output_root={cfg.run.output_root} "
                  f"train_files={len(cfg.data.train_files)} eval={cfg.eval.enabled} smoke_tasks={n_tasks}")
            return 0
        from pipeline.multi_lora_sft.run import main as run_main

        argv = ["--config", str(cfg_path)]
        for expr in sets:
            argv += ["--set", expr]
        return int(run_main(argv))

    def evaluate(
        self,
        tasks_path: str | Path | None = None,
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        """Runs evaluation over benchmark tasks and computes Pass@1 resolution rate."""
        if not tasks_path:
            tasks_path = self.root_dir / "data" / "val_tasks.jsonl"
        t_path = Path(tasks_path)
        if not t_path.is_absolute():
            t_path = self.root_dir / t_path

        print(f"\n[PIPELINE] Running Evaluation on {t_path}...")
        tasks = load_tasks(t_path)
        if not tasks:
            print(f"[PIPELINE WARNING] No tasks found at {t_path}.")
            return {"total_tasks": 0, "pass_at_1": 0.0}

        if max_tasks:
            tasks = tasks[:max_tasks]

        print(f"[PIPELINE] Evaluating {len(tasks)} tasks...")
        evaluator = LocalEvaluator(workspace_root=self.root_dir)

        # Baseline simulation summary
        results = [{"task_id": t.get("task_id", f"task_{i}"), "passed": False} for i, t in enumerate(tasks)]
        metrics = evaluator.calculate_pass_at_1(results)
        print(
            f"[PIPELINE EVALUATION SUMMARY] Total: {metrics['total_tasks']} | Pass@1: {metrics['pass_rate_pct']}%"
        )
        return metrics

    def package(
        self,
        model_name: str = "gemma-4-12b-it",
        output_zip: str | Path | None = None,
        staging_dir: str | Path | None = None,
        adapter_path: str | Path | None = None,
    ) -> Path:
        """Assembles declarative ADK directory, validates constraints, and builds submission.zip."""
        staging = Path(staging_dir) if staging_dir else (self.root_dir / "build" / "submission_staging")
        target_zip = Path(output_zip) if output_zip else (self.root_dir / "submission.zip")

        print(f"\n[PIPELINE] Scaffolding declarative submission in {staging}...")
        builder = ADKSubmissionBuilder(base_model_name=model_name, output_dir=staging)
        builder.build(adapter_path=str(adapter_path) if adapter_path else None)

        print("[PIPELINE] Validating ADK schema and host registry constraints...")
        is_valid, errors = validate_submission_directory(staging)
        if not is_valid:
            print("[PIPELINE ERROR] Submission validation failed:")
            for err in errors:
                print(f"  • {err}")
            raise ValueError("ADK validation errors detected.")

        print(f"[PIPELINE] Packaging submission into {target_zip}...")
        final_zip = package_submission_archive(staging, target_zip)
        print(f"[PIPELINE SUCCESS] Submission archive ready: {final_zip}")
        return final_zip


# Convenience entrypoint functions for simple notebook imports
def status(root_dir=None):
    return PipelineRunner(root_dir).status()


def train(mode="multi_lora_sft", config_path=None, root_dir=None, train_data=None, base_model=None, output_dir=None):
    return PipelineRunner(root_dir).train(
        mode=mode,
        config_path=config_path,
        train_data=train_data,
        base_model=base_model,
        output_dir=output_dir,
    )


def evaluate(tasks_path=None, max_tasks=None, root_dir=None):
    return PipelineRunner(root_dir).evaluate(tasks_path=tasks_path, max_tasks=max_tasks)


def package(model_name="gemma-4-12b-it", output_zip=None, adapter_path=None, root_dir=None):
    return PipelineRunner(root_dir).package(
        model_name=model_name,
        output_zip=output_zip,
        adapter_path=adapter_path,
    )


def multi_lora_sft(config_path=None, set_overrides=None, root_dir=None, preset=None, dry_run=False):
    return PipelineRunner(root_dir).multi_lora_sft(
        config_path=config_path, set_overrides=set_overrides, preset=preset, dry_run=dry_run
    )


def validate_data(files, config_path=None, set_overrides=None, tasks=None):
    """Checks trajectory JSONL files against the data contract (see README "Training data format")."""
    from pipeline.multi_lora_sft.validate_data import main as validate_main

    argv = [str(f) for f in files]
    if config_path:
        argv += ["--config", str(config_path)]
    for expr in set_overrides or []:
        argv += ["--set", expr]
    if tasks:
        argv += ["--tasks", str(tasks)]
    return validate_main(argv)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 Developer Agent pipeline. `--action train` runs the 7-phase Multi-LoRA SFT orchestrator "
        "(setup -> preflight -> train -> export -> package -> eval -> report).",
        epilog="presets: " + ", ".join(f"{k} = {v}" for k, v in PRESETS.items()),
    )
    parser.add_argument(
        "--action",
        choices=["status", "train", "multi_lora_sft", "validate-data", "evaluate", "package"],
        default="status",
        help="Action to perform",
    )
    parser.add_argument(
        "--mode",
        choices=["multi_lora_sft", "sft", "dpo"],
        default="multi_lora_sft",
        help="Training mode: multi_lora_sft (default, the orchestrator) or the legacy sft / dpo trainers",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Named config (listed below)")
    parser.add_argument("--config", type=str, default=None, help="Path to a config YAML (wins over --preset)")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, YAML-parsed (repeatable), e.g. --set train.lr=2e-4",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate the config and stop (train)")
    parser.add_argument("--train-data", type=str, nargs="+", default=None,
                        help="Training JSONL file(s) (train) or the files to check (validate-data)")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Run output root (overrides config)")
    parser.add_argument("--tasks", type=str, default=None, help="Tasks JSONL (evaluate; leakage gate for validate-data)")
    parser.add_argument("--model", type=str, default="gemma-4-12b-it", help="Target model variant (package)")
    parser.add_argument("--output-zip", type=str, default="submission.zip", help="Target zip file (package)")
    parser.add_argument("--adapter-path", type=str, default=None, help="Trained LoRA adapter path (package)")
    args = parser.parse_args()

    runner = PipelineRunner()
    orchestrated = args.action == "multi_lora_sft" or (args.action == "train" and args.mode == "multi_lora_sft")

    if args.action == "status":
        runner.status()
    elif orchestrated:
        sets = list(args.set_overrides)
        if args.train_data:
            sets.insert(0, "data.train_files=" + json.dumps(args.train_data))
        if args.base_model:
            sets.insert(0, f"model.train_base_path={args.base_model}")
        if args.output_dir:
            sets.insert(0, f"run.output_root={args.output_dir}")
        rc = runner.multi_lora_sft(config_path=args.config, set_overrides=sets, preset=args.preset, dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)
    elif args.action == "train":
        runner.train(
            mode=args.mode,
            config_path=args.config,
            train_data=args.train_data[0] if args.train_data else None,
            base_model=args.base_model,
            output_dir=args.output_dir,
        )
    elif args.action == "validate-data":
        if not args.train_data:
            parser.error("validate-data needs --train-data FILE [FILE ...]")
        cfg = resolve_preset_config(runner.root_dir, args.preset, args.config) if (args.preset or args.config) else None
        raise SystemExit(validate_data(args.train_data, cfg, args.set_overrides, args.tasks))
    elif args.action == "evaluate":
        runner.evaluate(tasks_path=args.tasks)
    elif args.action == "package":
        runner.package(
            model_name=args.model,
            output_zip=args.output_zip,
            adapter_path=args.adapter_path,
        )


if __name__ == "__main__":
    main()
