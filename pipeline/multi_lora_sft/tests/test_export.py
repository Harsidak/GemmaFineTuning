"""Unit tests for `pipeline.multi_lora_sft.phases.export` (design_doc.md §8.1, §11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import Gemma4ForConditionalGeneration

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.export import (
    EXPORT_KEY_RE,
    ExportValidationError,
    export_single_adapter,
)
from pipeline.multi_lora_sft.training.model import attach_lora_adapter


def test_export_matches_sample_key_format_bf16_rewrites_config_and_reloads(
    tmp_path: Path, repo_root: Path, tiny_gemma4_dir: Path
) -> None:
    # Verify sample submission adapter keys match our authoritative EXPORT_KEY_RE
    sample_st = (
        repo_root
        / "build"
        / "gemma-4-developer-agent"
        / "sample_submission"
        / "adapters"
        / "main_lora"
        / "adapter_model.safetensors"
    )
    if sample_st.is_file():
        sample_tensors = load_file(str(sample_st), device="cpu")
        assert sample_tensors
        assert all(EXPORT_KEY_RE.match(k) for k in sample_tensors)

    cfg = load_config(set_overrides=["lora.r=8", "lora.alpha=16"])
    eff = cfg.resolve_adapter("main_lora")

    base = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.float32)
    peft_model, target_regex, matched = attach_lora_adapter(base, eff.lora)

    # Simulate 1 training update so lora_B is non-zero
    for name, p in peft_model.named_parameters():
        if ".lora_B." in name:
            p.data.fill_(0.01)

    raw_dir = tmp_path / "raw_adapter"
    peft_model.save_pretrained(raw_dir)
    (raw_dir / "matched_target_modules.json").write_text(
        json.dumps({"target_regex": target_regex, "matched_modules": matched}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "exported" / "main_lora"
    info = export_single_adapter(
        adapter_name="main_lora",
        source_adapter_dir=raw_dir,
        output_adapter_dir=out_dir,
        served_model_id="gemma-4-e4b-it-qat-w4a16-ct",
        eff_adapter=eff,
    )
    assert info["r"] == 8
    assert info["base_model_name_or_path"] == "google/gemma-4-e4b-it-qat-w4a16-ct"

    exported_tensors = load_file(str(out_dir / "adapter_model.safetensors"), device="cpu")
    assert all(t.dtype == torch.bfloat16 for t in exported_tensors.values())
    assert all(EXPORT_KEY_RE.match(k) for k in exported_tensors)

    exported_cfg = json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert exported_cfg["inference_mode"] is True
    assert exported_cfg["base_model_name_or_path"] == "google/gemma-4-e4b-it-qat-w4a16-ct"
    assert isinstance(exported_cfg["target_modules"], list)
    assert exported_cfg["target_modules"] == sorted(matched)

    # Reload onto a fresh tiny Gemma 4 WITH vision tower and verify PEFT attaches cleanly without touching vision!
    fresh_base = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.bfloat16)
    reloaded = PeftModel.from_pretrained(fresh_base, str(out_dir))
    lora_param_names = [n for n, _ in reloaded.named_parameters() if "lora_" in n]
    assert lora_param_names
    assert all("vision_tower" not in n and "language_model" in n for n in lora_param_names)


def test_export_rejects_planted_vision_key_nan_and_all_zero_b(tmp_path: Path) -> None:
    src = tmp_path / "bad_src"
    src.mkdir(parents=True)
    (src / "adapter_config.json").write_text(
        json.dumps({"r": 4, "lora_alpha": 8, "peft_type": "LORA", "task_type": "CAUSAL_LM"}),
        encoding="utf-8",
    )

    # 1. Planted vision key -> rejected
    save_file(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((4, 16)),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((16, 4)),
            "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((4, 16)),
            "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((16, 4)),
        },
        str(src / "adapter_model.safetensors"),
    )
    with pytest.raises(ExportValidationError, match="multimodal"):
        export_single_adapter(
            adapter_name="main_lora",
            source_adapter_dir=src,
            output_adapter_dir=tmp_path / "out1",
            served_model_id="gemma-4-31b-it-qat-w4a16-ct",
        )

    # 2. All-zero B matrices -> rejected
    save_file(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((4, 16)),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((16, 4)),
        },
        str(src / "adapter_model.safetensors"),
    )
    with pytest.raises(ExportValidationError, match="identically zero"):
        export_single_adapter(
            adapter_name="main_lora",
            source_adapter_dir=src,
            output_adapter_dir=tmp_path / "out2",
            served_model_id="gemma-4-31b-it-qat-w4a16-ct",
        )

    # 3. NaN weight -> rejected
    bad_b = torch.ones((16, 4))
    bad_b[0, 0] = float("nan")
    save_file(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((4, 16)),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": bad_b,
        },
        str(src / "adapter_model.safetensors"),
    )
    with pytest.raises(ExportValidationError, match="NaN or Inf"):
        export_single_adapter(
            adapter_name="main_lora",
            source_adapter_dir=src,
            output_adapter_dir=tmp_path / "out3",
            served_model_id="gemma-4-31b-it-qat-w4a16-ct",
        )
