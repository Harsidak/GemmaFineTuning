import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Standard target modules for Gemma / Gemma 2 / Gemma 4 architectures
GEMMA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

def get_gemma_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none"
) -> Any:
    """Creates a PEFT LoraConfig specifically tuned for Gemma 4."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as e:
        logger.error(f"peft not installed: {e}")
        raise

    targets = target_modules or GEMMA_TARGET_MODULES

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=targets,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )
    logger.info(f"Initialized LoraConfig: r={r}, alpha={lora_alpha}, targets={targets}")
    return config


def apply_lora_to_model(model: Any, lora_config: Any, prepare_for_kbit: bool = True) -> Any:
    """Prepares model for k-bit training and attaches LoRA adapters."""
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        logger.error(f"peft not installed: {e}")
        raise

    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
