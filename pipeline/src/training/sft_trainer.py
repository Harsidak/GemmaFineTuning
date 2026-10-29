import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..data.loader import load_trajectories
from ..data.processor import format_trajectory_for_sft
from ..models.loader import load_gemma_model_and_tokenizer
from ..models.lora import get_gemma_lora_config, apply_lora_to_model

logger = logging.getLogger(__name__)

def run_sft_training(config_path: str | Path):
    """Executes SFT fine-tuning using the specified YAML config."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("dataset", {})

    logger.info("Initializing Model and Tokenizer...")
    model, tokenizer = load_gemma_model_and_tokenizer(
        model_name_or_path=model_cfg.get("base_model_name_or_path", "google/gemma-4-12b-it"),
        use_4bit=model_cfg.get("use_4bit_quantization", True),
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16")
    )

    logger.info("Configuring LoRA...")
    lora_config = get_gemma_lora_config(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules")
    )
    model = apply_lora_to_model(model, lora_config)

    logger.info("Loading trajectories...")
    train_trajectories = load_trajectories(data_cfg.get("train_data_path", "data/train_trajectories.jsonl"))

    if not train_trajectories:
        logger.warning("No training trajectories found. Please populate data/train_trajectories.jsonl.")
        return

    # Process into text records
    formatted_texts = [
        format_trajectory_for_sft(t.get("messages", []))
        for t in train_trajectories
    ]

    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        logger.error(f"TRL or Datasets not installed: {e}")
        raise

    train_dataset = Dataset.from_dict({"text": formatted_texts})

    training_args = SFTConfig(
        output_dir=train_cfg.get("output_dir", "models/checkpoints/gemma4_lora_sft"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        bf16=train_cfg.get("bf16", True),
        max_seq_length=data_cfg.get("max_seq_length", 8192),
        packing=data_cfg.get("packing", False),
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args
    )

    logger.info("Starting SFT Training loop...")
    trainer.train()

    # Save final LoRA adapter
    final_output_path = Path(train_cfg.get("output_dir")) / "final_adapter"
    trainer.model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    logger.info(f"LoRA adapter successfully saved to {final_output_path}")
