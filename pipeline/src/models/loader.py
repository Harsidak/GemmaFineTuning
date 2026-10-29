import logging
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)

def load_gemma_model_and_tokenizer(
    model_name_or_path: str = "google/gemma-4-12b-it",
    use_4bit: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """Loads Gemma 4 model and tokenizer with optional 4-bit quantization (QLoRA).

    Compatible with Gemma 4 12B, 31B QAT, and 26B MoE architectures.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        logger.error(f"Missing required ML libraries: {e}")
        raise

    logger.info(f"Loading tokenizer for: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        compute_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        logger.info("Configured BitsAndBytes 4-bit NF4 Quantization (QLoRA).")

    logger.info(f"Loading Gemma model weights: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=getattr(torch, torch_dtype, torch.bfloat16),
        trust_remote_code=True
    )

    return model, tokenizer
