#!/usr/bin/env python3
"""CLI runner for fine-tuning Gemma 4 developer agent models."""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.src.training.sft_trainer import run_sft_training
from pipeline.src.training.dpo_trainer import run_dpo_training

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Developer Agent Training Runner")
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo"],
        default="sft",
        help="Training mode: Supervised Fine-Tuning (sft) or Direct Preference Optimization (dpo)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline/configs/train_lora_sft.yaml",
        help="Path to training YAML configuration file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.mode == "sft":
        print(f"--- Launching SFT Training with config: {args.config} ---")
        run_sft_training(args.config)
    elif args.mode == "dpo":
        print(f"--- Launching DPO Training with config: {args.config} ---")
        run_dpo_training(args.config)

if __name__ == "__main__":
    main()
