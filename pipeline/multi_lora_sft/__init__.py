"""Time-budgeted Multi-LoRA SFT pipeline for the Kaggle Gemma 4 Developer Agent competition.

Entrypoint:
    python -m pipeline.multi_lora_sft.run --config <yaml> [--set key=value ...]
"""

from .config import ConfigError, PipelineConfig, load_config

__all__ = ["ConfigError", "PipelineConfig", "load_config"]
