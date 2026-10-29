#!/usr/bin/env python3
"""CLI runner to build declarative submission scaffolds and package submission.zip."""

import argparse
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.src.agent.builder import ADKSubmissionBuilder
from pipeline.src.utils.packager import package_submission_archive, validate_submission_directory

def main():
    parser = argparse.ArgumentParser(description="Package ADK Agent Submission for Kaggle")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4-12b-it",
        help="Base Gemma 4 model variant (e.g. gemma-4-12b-it, gemma-4-31b-it-qat-w4a16-ct)"
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default="build/submission_staging",
        help="Directory to scaffold declarative files before zipping"
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default="submission.zip",
        help="Output submission zip path"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained LoRA adapter directory (optional)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

    print(f"Building declarative scaffold in: {args.staging_dir}")
    builder = ADKSubmissionBuilder(
        base_model_name=args.model,
        output_dir=args.staging_dir
    )
    staging_path = builder.build(adapter_path=args.adapter_path)

    print("Validating declarative submission structure...")
    is_valid, errors = validate_submission_directory(staging_path)
    if not is_valid:
        print("Validation FAILED with errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Validation PASSED.")
    print(f"Compressing into {args.output_zip}...")
    zip_path = package_submission_archive(staging_path, args.output_zip)
    print(f"Done! Ready to upload to Kaggle: {zip_path.resolve()}")

if __name__ == "__main__":
    main()
