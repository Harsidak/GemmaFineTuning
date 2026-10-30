"""Unit tests for LoRA target regex and KV-shared layer exclusion (design_doc.md §3.4, §7.4, §11)."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Gemma4ForConditionalGeneration

from pipeline.multi_lora_sft.config import LoraSection
from pipeline.multi_lora_sft.training.model import (
    attach_lora_adapter,
    get_kv_shared_layer_indices,
)


def test_lora_targets_language_model_only_and_excludes_kv_shared_kv(tiny_gemma4_dir: Path) -> None:
    # Load WITH vision_tower intact to prove regex NEVER touches vision_tower even when present in memory!
    model = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.bfloat16)
    assert getattr(model.model, "vision_tower", None) is not None

    # Tiny model has 4 text layers with num_kv_shared_layers=2 -> layers [2, 3] are KV-shared
    shared = get_kv_shared_layer_indices(model.config)
    assert shared == [2, 3]

    lora_cfg = LoraSection(r=8, alpha=16, exclude_kv_shared=True)
    peft_model, _, matched = attach_lora_adapter(model, lora_cfg)

    assert len(matched) > 0
    for mod_name in matched:
        assert "language_model" in mod_name
        assert "vision_tower" not in mod_name
        assert "audio_tower" not in mod_name

    # Layers 0 and 1 have q, k, v, o, gate, up, down; layers 2 and 3 exclude k_proj and v_proj!
    assert "model.language_model.layers.0.self_attn.k_proj" in matched
    assert "model.language_model.layers.1.self_attn.v_proj" in matched
    assert "model.language_model.layers.2.self_attn.q_proj" in matched
    assert "model.language_model.layers.2.self_attn.o_proj" in matched
    assert "model.language_model.layers.2.self_attn.k_proj" not in matched
    assert "model.language_model.layers.2.self_attn.v_proj" not in matched
    assert "model.language_model.layers.3.self_attn.k_proj" not in matched
    assert "model.language_model.layers.3.self_attn.v_proj" not in matched

    # Double check trainable parameter names on PeftModel
    trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("vision_tower" not in n and "language_model" in n for n in trainable_names)
