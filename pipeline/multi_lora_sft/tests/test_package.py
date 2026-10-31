"""Unit tests for `pipeline.multi_lora_sft.phases.package`, `evaluate`, and `parity` (§8.2, §9, §11)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.evaluate import (
    clopper_pearson_interval,
    format_smoke_eval_banner,
)
from pipeline.multi_lora_sft.phases.package import (
    SubmissionValidationError,
    run_package_phase,
    validate_staged_submission,
)
from pipeline.multi_lora_sft.runctx import RunContext
from pipeline.multi_lora_sft.training.parity import evaluate_parity_against_hf


def _make_fake_exported_adapter(target_dir: Path, r: int = 8) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                (r, 16), dtype=torch.bfloat16
            ),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                (16, r), dtype=torch.bfloat16
            ),
        },
        str(target_dir / "adapter_model.safetensors"),
    )
    (target_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "google/gemma-4-e4b-it-qat-w4a16-ct",
                "inference_mode": True,
                "r": r,
                "lora_alpha": r * 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["model.language_model.layers.0.self_attn.q_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target_dir


def test_package_sample_template_rewrites_model_preserves_include_and_validates(
    tmp_path: Path, repo_root: Path
) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    work_zip = tmp_path / "working" / "submission.zip"
    work_zip.parent.mkdir(parents=True)

    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=pkg_test",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            f"submission.working_zip_path={work_zip.as_posix()}",
        ]
    )
    ctx = RunContext.create(cfg, t0=1000.0)
    cfg.save_yaml(ctx.resolved_config_path)

    _make_fake_exported_adapter(ctx.adapters_dir / "main_lora", r=16)
    _make_fake_exported_adapter(ctx.adapters_dir / "tool_lora", r=8)

    st = run_package_phase(ctx, cfg=cfg)
    assert st["state"] == "ok"
    assert ctx.submission_zip_path.is_file()
    assert work_zip.is_file()

    # `!include` tags preserved verbatim while `model:` was rewritten to gemma-4-e4b-it-qat-w4a16-ct
    agent_txt = (ctx.submission_dir / "agent.yaml").read_text(encoding="utf-8")
    sub_txt = (ctx.submission_dir / "sub_agents" / "code_analyzer.yaml").read_text(encoding="utf-8")
    assert "model: gemma-4-e4b-it-qat-w4a16-ct" in agent_txt
    assert "model: gemma-4-e4b-it-qat-w4a16-ct" in sub_txt
    assert "!include prompts/system.md" in agent_txt
    assert "!include ../prompts/analyzer.md" in sub_txt

    with zipfile.ZipFile(ctx.submission_zip_path) as zf:
        names = set(zf.namelist())
        assert "agent.yaml" in names
        assert "adapters/main_lora/adapter_model.safetensors" in names
        assert "adapters/tool_lora/adapter_model.safetensors" in names


@pytest.mark.parametrize("trained", [True, False])
def test_package_without_any_sample_template_still_ships_the_trained_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained: bool
) -> None:
    """Kaggle case: build/ is not in the repo dataset, so there is no sample template to copy. The scaffolded template
    must declare the adapter slots (else trained adapters never reach the zip) and must not leave empty slot folders
    (else validation fails and NO zip is written). Regression for the 2026-10-30 Kaggle run that ended without a zip."""
    import pipeline.multi_lora_sft.phases.package as pkg

    monkeypatch.setattr(pkg, "default_sample_template_dir", lambda: None)
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            f"run.run_id=pkg_no_template_{int(trained)}",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            "submission.template_dir=null",
            "submission.working_zip_path=null",
        ]
    )
    ctx = RunContext.create(cfg, t0=1000.0)
    cfg.save_yaml(ctx.resolved_config_path)
    if trained:
        _make_fake_exported_adapter(ctx.adapters_dir / "main_lora", r=16)
        _make_fake_exported_adapter(ctx.adapters_dir / "tool_lora", r=8)

    st = run_package_phase(ctx, cfg=cfg)
    assert st["state"] == "ok", st.get("error")
    assert ctx.submission_zip_path.is_file()
    agent_txt = (ctx.submission_dir / "agent.yaml").read_text(encoding="utf-8")
    sub_txt = (ctx.submission_dir / "sub_agents" / "code_analyzer.yaml").read_text(encoding="utf-8")
    with zipfile.ZipFile(ctx.submission_zip_path) as zf:
        names = set(zf.namelist())
    if trained:
        assert "adapter: main_lora" in agent_txt and "adapter: tool_lora" in sub_txt
        assert {"adapters/main_lora/adapter_model.safetensors", "adapters/tool_lora/adapter_model.safetensors"} <= names
        assert st["outputs"]["adapter_sources"] == {"main_lora": "exported:main_lora", "tool_lora": "exported:tool_lora"}
    else:
        assert "adapter:" not in agent_txt and "adapter:" not in sub_txt
        assert not any(n.startswith("adapters/") for n in names)
        assert not (ctx.submission_dir / "adapters").exists()


def test_package_fallback_order_checkpoint_fallback_dir_and_bare_model(
    tmp_path: Path, repo_root: Path
) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    fb_dir = tmp_path / "fallback_adapters"
    _make_fake_exported_adapter(fb_dir / "main_lora", r=8)
    # Note: tool_lora is missing EVERYWHERE -> must fall back to bare base model (stripping `adapter: tool_lora`)!

    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=pkg_fallback",
            "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            f"submission.fallback_adapters_dir={fb_dir.as_posix()}",
            "submission.working_zip_path=null",
        ]
    )
    ctx = RunContext.create(cfg, t0=1000.0)
    cfg.save_yaml(ctx.resolved_config_path)

    st = run_package_phase(ctx, cfg=cfg)
    assert st["state"] == "ok"
    sources = st["outputs"]["adapter_sources"]
    assert sources["main_lora"].startswith("fallback_dir:")
    assert sources["tool_lora"] == "bare_base_model"

    sub_txt = (ctx.submission_dir / "sub_agents" / "code_analyzer.yaml").read_text(encoding="utf-8")
    assert "adapter:" not in sub_txt
    assert "!include ../prompts/analyzer.md" in sub_txt


def test_package_catches_extension_traversal_token_and_size_violations(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=pkg_bad",
            "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
        ]
    )
    ctx = RunContext.create(cfg, t0=1000.0)
    cfg.save_yaml(ctx.resolved_config_path)
    run_package_phase(ctx, cfg=cfg)  # builds clean baseline staging directory

    # 1. Root Python file rejected
    rogue_py = ctx.submission_dir / "agent_entrypoint.py"
    rogue_py.write_text("print('hack')\n", encoding="utf-8")
    with pytest.raises(SubmissionValidationError):
        validate_staged_submission(ctx.submission_dir)
    rogue_py.unlink()

    # 2. Disallowed pickle weight extension (.pt) rejected
    rogue_pt = ctx.submission_dir / "weights.pt"
    rogue_pt.write_bytes(b"bad")
    with pytest.raises(SubmissionValidationError, match="Disallowed file extension"):
        validate_staged_submission(ctx.submission_dir)
    rogue_pt.unlink()

    # 3. `!include` escaping submission root rejected
    orig_agent = (ctx.submission_dir / "agent.yaml").read_text(encoding="utf-8")
    (ctx.submission_dir / "agent.yaml").write_text(
        orig_agent.replace("!include prompts/system.md", "!include ../../outside.md"),
        encoding="utf-8",
    )
    with pytest.raises(SubmissionValidationError, match="escapes submission root"):
        validate_staged_submission(ctx.submission_dir)
    (ctx.submission_dir / "agent.yaml").write_text(orig_agent, encoding="utf-8")

    # 4. Token ceiling > 32768 rejected
    orig_samp = (ctx.submission_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8")
    (ctx.submission_dir / "configs" / "sampling.yaml").write_text(
        "max_output_tokens: 65536\n", encoding="utf-8"
    )
    with pytest.raises(SubmissionValidationError, match="max_output_tokens"):
        validate_staged_submission(ctx.submission_dir)
    (ctx.submission_dir / "configs" / "sampling.yaml").write_text(orig_samp, encoding="utf-8")

    # 5. Total size >= 3 GiB rejected
    import pipeline.multi_lora_sft.phases.package as pkg_mod

    monkeypatch.setattr(pkg_mod, "MAX_SUBMISSION_BYTES", 100)
    with pytest.raises(SubmissionValidationError, match="exceeds 3 GiB limit"):
        validate_staged_submission(ctx.submission_dir)


def test_smoke_eval_clopper_pearson_and_parity_verdict() -> None:
    lo, hi = clopper_pearson_interval(1, 3, alpha=0.05)
    assert round(lo * 100, 1) == 0.8
    assert round(hi * 100, 1) == 90.6

    hf_ref = {
        "adapter": "main_lora",
        "prompts": [
            {
                "with_adapter": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2}],
                "without_adapter": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2}],
            }
        ],
    }
    ok_res = evaluate_parity_against_hf(
        hf_ref,
        [[{"token": "a", "logprob": -0.11, "top_logprobs": {"a": -0.11}}, {"token": "b", "logprob": -0.21, "top_logprobs": {"b": -0.21}}]],
        [[{"token": "a", "logprob": -0.10, "top_logprobs": {"a": -0.10}}, {"token": "b", "logprob": -0.20, "top_logprobs": {"b": -0.20}}]],
    )
    assert ok_res["verdict"] == "OK"

    suspect_res = evaluate_parity_against_hf(
        hf_ref,
        [[{"token": "WRONG", "logprob": -0.5, "top_logprobs": {"a": -2.0}}, {"token": "b", "logprob": -0.2, "top_logprobs": {}}]],
        [[{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -0.2}]],
    )
    assert suspect_res["verdict"] == "SUSPECT"
    assert "r=16" in suspect_res["recommendation"]

    banner = format_smoke_eval_banner(
        served_model_id="gemma-4-31b-it-qat-w4a16-ct",
        adapter_ranks={"main_lora": 32},
        run_id="test_run",
        task_rows=[
            {"task_id": "fastapi_15661", "result": "PASS", "patch_applied": True, "turns": 23, "tool_call_errors": 0, "duration_sec": 702},
            {"task_id": "django_2", "result": "FAIL", "patch_applied": True, "turns": 31, "tool_call_errors": 2, "duration_sec": 845},
            {"task_id": "sympy_3", "result": "TIMEOUT", "patch_applied": False, "turns": None, "tool_call_errors": None, "duration_sec": 900},
        ],
        parity_results={"main_lora": ok_res},
    )
    assert "SCORE: 1/3 = 33.3%   (95% Clopper-Pearson interval: 0.8%–90.6%)" in banner
