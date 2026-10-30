"""Unit tests for `pipeline.multi_lora_sft.config` (design_doc.md §10.1, §11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.multi_lora_sft.config import ConfigError, PipelineConfig, load_config


def test_defaults_and_all_repo_configs_load(repo_root: Path) -> None:
    default_cfg = load_config()
    assert isinstance(default_cfg, PipelineConfig)
    assert default_cfg.model.served_model_id == "gemma-4-31b-it-qat-w4a16-ct"
    assert [a.name for a in default_cfg.adapters] == ["main_lora", "tool_lora"]

    configs_dir = repo_root / "pipeline" / "multi_lora_sft" / "configs"
    for name in ("base.yaml", "e4b_calibration.yaml", "e4b_full.yaml", "31b_full.yaml"):
        cfg = load_config(configs_dir / name)
        assert cfg.effective_served_model_id() in {
            "gemma-4-e4b-it-qat-w4a16-ct",
            "gemma-4-31b-it-qat-w4a16-ct",
        }


def test_unknown_keys_rejected_in_yaml_and_set(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("train:\n  nonexistent_knob: 123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        load_config(bad_yaml)

    with pytest.raises(ConfigError, match="Unknown configuration key"):
        load_config(set_overrides=["lora.not_a_field=16"])

    with pytest.raises(ConfigError, match="unknown section"):
        load_config(
            set_overrides=[
                'adapters=[{name: main_lora, overrides: {bogus_section: {x: 1}}}]'
            ]
        )


def test_set_overrides_and_per_adapter_deep_merge(repo_root: Path) -> None:
    cfg = load_config(
        repo_root / "pipeline" / "multi_lora_sft" / "configs" / "31b_full.yaml",
        set_overrides=[
            "run.mode=calibration",
            "run.calibration_minutes=25",
            "lora.r=32",
            "data.train_files=[a.jsonl, b.jsonl]",
            "data.template_kwargs={enable_thinking: true}",
        ],
    )
    assert cfg.run.mode == "calibration"
    assert cfg.run.calibration_minutes == 25.0
    assert cfg.data.train_files == ["a.jsonl", "b.jsonl"]
    assert cfg.data.template_kwargs == {"enable_thinking": True}

    eff_main = cfg.resolve_adapter("main_lora")
    eff_tool = cfg.resolve_adapter("tool_lora")
    assert eff_main.lora.r == 32
    assert eff_main.train.max_seq_length == 16384
    assert eff_tool.lora.r == 16
    assert eff_tool.lora.alpha == 32
    assert eff_tool.train.max_seq_length == 8192
    # Global config remains unmutated
    assert cfg.lora.r == 32
    assert cfg.train.max_seq_length == 16384


def test_invalid_served_model_rank_and_adapter_names_fail() -> None:
    with pytest.raises(ConfigError, match="Invalid served model"):
        load_config(set_overrides=["model.served_model_id=gemma-4-31b-it"])

    with pytest.raises(ConfigError, match="1..128"):
        load_config(set_overrides=["lora.r=256"])

    with pytest.raises(ConfigError, match="1..128"):
        load_config(
            set_overrides=[
                "adapters=[{name: main_lora, overrides: {lora: {r: 129}}}]"
            ]
        )

    with pytest.raises(ConfigError, match="Invalid adapter name"):
        load_config(set_overrides=["adapters=[{name: '../bad_name'}]"])

    with pytest.raises(ConfigError, match="multimodal module"):
        load_config(set_overrides=[r"lora.target_regex='.*\.(q_proj|v_proj)$'"])
