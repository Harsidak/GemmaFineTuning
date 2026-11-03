"""Checks training-trajectory JSONL against the pipeline's data contract (no GPU needed).

    uv run python scripts/validate_dataset.py data/my_trajectories/*.jsonl [--config <yaml>] [--tasks <tasks.jsonl>]

The checks live in `pipeline/multi_lora_sft/validate_data.py`; the record format is in README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.multi_lora_sft.validate_data import main

if __name__ == "__main__":
    raise SystemExit(main())
