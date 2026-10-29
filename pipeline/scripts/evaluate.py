#!/usr/bin/env python3
"""CLI runner for evaluating agent configurations on benchmark tasks."""

import argparse
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.src.data.loader import load_tasks
from pipeline.src.eval.harness import LocalEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate Developer Agent locally")
    parser.add_argument(
        "--tasks",
        type=str,
        default="data/val_tasks.jsonl",
        help="Path to tasks.jsonl benchmark file"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks to evaluate (defaults to all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

    tasks = load_tasks(args.tasks)
    if not tasks:
        print(f"No tasks loaded from {args.tasks}. Ensure file exists.")
        return

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"Loaded {len(tasks)} tasks for evaluation.")
    evaluator = LocalEvaluator()

    # Evaluator summary simulation
    print("Local evaluation harness ready. Execute against your configured sandbox.")

if __name__ == "__main__":
    main()
