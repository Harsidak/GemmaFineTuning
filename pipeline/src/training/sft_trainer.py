import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

from ..data.loader import load_trajectories
from ..data.processor import (
    TemplateMismatchError,
    default_tools,
    load_chat_template,
    tokenize_trajectory,
)
from ..models.loader import load_gemma_model_and_tokenizer
from ..models.lora import apply_lora_to_model, get_gemma_lora_config

logger = logging.getLogger(__name__)

# Config keys renamed or removed in current transformers/TRL, mapped to their replacements.
_RENAMED_TRAINING_KEYS = {
    "warmup_ratio": "warmup_steps",  # warmup_steps takes a float in [0, 1) as a ratio
    "evaluation_strategy": "eval_strategy",
}


def _build_sft_config(train_cfg: dict[str, Any], max_length: int, packing: bool, output_dir: str) -> Any:
    """Builds SFTConfig from the YAML `training:` section, skipping keys this TRL version lacks."""
    from trl import SFTConfig

    known = {f.name for f in dataclasses.fields(SFTConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in train_cfg.items():
        key = _RENAMED_TRAINING_KEYS.get(key, key)
        if key not in known:
            logger.warning(f"Ignoring training.{key}: not an SFTConfig field in this TRL version")
            continue
        kwargs[key] = value
    if "learning_rate" in kwargs:
        kwargs["learning_rate"] = float(kwargs["learning_rate"])
    # No eval set is built here, so an eval strategy would fail at init.
    kwargs.pop("eval_strategy", None)
    kwargs.setdefault("report_to", "none")
    if kwargs.get("gradient_checkpointing"):
        kwargs.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    kwargs.update(output_dir=output_dir, max_length=max_length, packing=packing)
    return SFTConfig(**kwargs)


def run_sft_training(
    config_path: str | Path,
    train_data_path: str | Path | None = None,
    base_model: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    """Executes SFT fine-tuning using the specified YAML config.

    train_data_path, base_model and output_dir override the config, so private data and attached
    models are passed in per run instead of being written into the public repo.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = dict(cfg.get("training", {}))
    data_cfg = cfg.get("dataset", {})

    base_model = str(base_model or model_cfg.get("base_model_name_or_path"))
    train_data_path = train_data_path or data_cfg.get("train_data_path", "data/train_trajectories.jsonl")
    cfg_output_dir = train_cfg.pop("output_dir", "models/checkpoints/gemma4_lora_sft")
    output_dir = str(output_dir or cfg_output_dir)
    max_length = int(data_cfg.get("max_seq_length", 32768))

    logger.info("Loading trajectories...")
    train_trajectories = load_trajectories(train_data_path)
    if not train_trajectories:
        logger.warning(f"No training trajectories found at {train_data_path}.")
        return

    logger.info("Initializing Model and Tokenizer...")
    model, tokenizer = load_gemma_model_and_tokenizer(
        model_name_or_path=base_model,
        use_4bit=model_cfg.get("use_4bit_quantization", True),
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16")
    )

    # Render with the template of the model that will be served; the training checkpoint may ship another.
    chat_template = load_chat_template(data_cfg.get("chat_template_path"))
    template_kwargs = data_cfg.get("chat_template_kwargs") or {}

    records, dropped_long, dropped_template, no_targets = [], 0, 0, 0
    for t in train_trajectories:
        tools = t.get("tools") or default_tools()
        try:
            rec = tokenize_trajectory(
                t.get("messages", []), tokenizer, tools=tools, chat_template=chat_template,
                max_length=max_length, **template_kwargs,
            )
        except TemplateMismatchError as e:
            dropped_template += 1
            logger.warning(f"Skipping {t.get('task_id', '?')}: {e}")
            continue
        if rec is None:
            dropped_long += 1
        elif all(label == -100 for label in rec["labels"]):
            no_targets += 1
        else:
            records.append(rec)
    logger.info(
        f"Tokenized {len(records)}/{len(train_trajectories)} trajectories "
        f"(over {max_length} tokens: {dropped_long}, template mismatch: {dropped_template}, "
        f"no assistant tokens: {no_targets})"
    )
    if not records:
        raise ValueError("No trainable trajectories after tokenization.")

    logger.info("Configuring LoRA...")
    lora_config = get_gemma_lora_config(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules")
    )
    model = apply_lora_to_model(model, lora_config)

    try:
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"TRL or Datasets not installed: {e}")
        raise

    # Pre-tokenized with a "labels" column: TRL skips its own templating and keeps our assistant mask.
    train_dataset = Dataset.from_list(records)
    training_args = _build_sft_config(train_cfg, max_length, data_cfg.get("packing", False), output_dir)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args
    )

    logger.info("Starting SFT Training loop...")
    trainer.train()

    # Save final LoRA adapter
    final_output_path = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    logger.info(f"LoRA adapter successfully saved to {final_output_path}")
    return final_output_path
