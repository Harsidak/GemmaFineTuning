"""Unified Pipeline Orchestrator for Gemma 4 Developer Agent.

This module acts as the central control plane for:
1. Training (SFT and DPO preference fine-tuning)
2. Local evaluation & SWE-bench sandbox testing
3. Declarative ADK submission packaging & validation
4. Workspace status monitoring

Can be imported directly into Kaggle launcher notebooks or executed via CLI:
    python pipeline/main.py --action [status | train | evaluate | package]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure project root or current pipeline directory is in sys.path
CURRENT_FILE = Path(__file__).resolve()
PIPELINE_DIR = CURRENT_FILE.parent
PROJECT_ROOT = PIPELINE_DIR.parent

for p in [str(PROJECT_ROOT), str(PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.src.agent.builder import ADKSubmissionBuilder
from pipeline.src.utils.packager import package_submission_archive, validate_submission_directory
from pipeline.src.data.loader import load_tasks, load_trajectories
from pipeline.src.eval.harness import LocalEvaluator

logger = logging.getLogger("PipelineRunner")

class PipelineRunner:
    """Modular coordinator for all Gemma fine-tuning and agent evaluation workflows."""

    def __init__(self, root_dir: Optional[str | Path] = None):
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT
        self.pipeline_dir = self.root_dir / "pipeline" if (self.root_dir / "pipeline").exists() else self.root_dir

    def status(self) -> Dict[str, Any]:
        """Inspects and returns the workspace readiness status."""
        rules_path = self.root_dir / ".agents" / "rules.md"
        val_tasks = self.root_dir / "data" / "val_tasks.jsonl"
        train_traj = self.root_dir / "data" / "train_trajectories.jsonl"
        submission_zip = self.root_dir / "submission.zip"
        mistakes_log = self.root_dir / "planning" / "MISTAKES.md"
        roadmap = self.root_dir / "planning" / "ROADMAP.md"
        resources_index = self.root_dir / "resources" / "INDEX.md"
        scripts_dir = self.root_dir / "scripts"
        configs = list((self.pipeline_dir / "configs").glob("*.yaml")) if (self.pipeline_dir / "configs").exists() else []

        status_info = {
            "root_dir": str(self.root_dir),
            "rules_ready": rules_path.exists(),
            "validation_tasks_found": val_tasks.exists(),
            "train_trajectories_found": train_traj.exists(),
            "configs_count": len(configs),
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
        print(f"• Config Files Available:    {len(configs)}")
        print(f"• Submission Archive:        {'[READY]' if status_info['packaged_submission_ready'] else '[NOT BUILT]'}")
        print("=" * 65)
        return status_info

    def train(
        self,
        mode: str = "sft",
        config_path: Optional[str | Path] = None,
        train_data: Optional[str | Path] = None,
        base_model: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ):
        """Runs Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO).

        train_data, base_model and output_dir override the config (SFT only), so private datasets and
        attached models are passed per run instead of being hardcoded in the public repo.
        """
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
        elif mode == "dpo":
            from pipeline.src.training.dpo_trainer import run_dpo_training
            run_dpo_training(cfg_path)
        else:
            raise ValueError(f"Unknown training mode: {mode}. Choose 'sft' or 'dpo'.")

    def evaluate(
        self,
        tasks_path: Optional[str | Path] = None,
        max_tasks: Optional[int] = None
    ) -> Dict[str, Any]:
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
        print(f"[PIPELINE EVALUATION SUMMARY] Total: {metrics['total_tasks']} | Pass@1: {metrics['pass_rate_pct']}%")
        return metrics

    def package(
        self,
        model_name: str = "gemma-4-12b-it",
        output_zip: Optional[str | Path] = None,
        staging_dir: Optional[str | Path] = None,
        adapter_path: Optional[str | Path] = None
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

def train(mode="sft", config_path=None, root_dir=None, train_data=None, base_model=None, output_dir=None):
    return PipelineRunner(root_dir).train(
        mode=mode, config_path=config_path, train_data=train_data, base_model=base_model, output_dir=output_dir
    )

def evaluate(tasks_path=None, max_tasks=None, root_dir=None):
    return PipelineRunner(root_dir).evaluate(tasks_path=tasks_path, max_tasks=max_tasks)

def package(model_name="gemma-4-12b-it", output_zip=None, adapter_path=None, root_dir=None):
    return PipelineRunner(root_dir).package(
        model_name=model_name,
        output_zip=output_zip,
        adapter_path=adapter_path
    )


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Developer Agent Pipeline Controller")
    parser.add_argument(
        "--action",
        choices=["status", "train", "evaluate", "package"],
        default="status",
        help="Action to perform"
    )
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft", help="Training mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--train-data", type=str, default=None, help="Training JSONL (overrides config)")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path or id (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint output dir (overrides config)")
    parser.add_argument("--tasks", type=str, default=None, help="Tasks JSONL file for eval")
    parser.add_argument("--model", type=str, default="gemma-4-12b-it", help="Target model variant")
    parser.add_argument("--output-zip", type=str, default="submission.zip", help="Target zip file")
    parser.add_argument("--adapter-path", type=str, default=None, help="Trained LoRA adapter path")
    args = parser.parse_args()

    runner = PipelineRunner()

    if args.action == "status":
        runner.status()
    elif args.action == "train":
        runner.train(
            mode=args.mode,
            config_path=args.config,
            train_data=args.train_data,
            base_model=args.base_model,
            output_dir=args.output_dir,
        )
    elif args.action == "evaluate":
        runner.evaluate(tasks_path=args.tasks)
    elif args.action == "package":
        runner.package(
            model_name=args.model,
            output_zip=args.output_zip,
            adapter_path=args.adapter_path
        )

if __name__ == "__main__":
    main()
