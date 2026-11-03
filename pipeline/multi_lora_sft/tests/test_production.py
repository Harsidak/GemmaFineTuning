"""Production hardening: local path defaults, step-based validation, report insights (curves, eval metrics, diagnosis)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.multi_lora_sft.config import (
    load_config,
    local_default_paths,
    resolve_default_paths,
    running_on_kaggle,
)
from pipeline.multi_lora_sft.phases.report_insights import (
    diagnosis,
    eval_metrics,
    learning_curve_lines,
    val_rows,
)
from pipeline.multi_lora_sft.runctx import PhaseStatus, RunContext


def test_kaggle_paths_fall_back_to_repo_paths_off_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLSFT_PLATFORM", "local")
    assert running_on_kaggle() is False
    cfg = load_config()
    local = local_default_paths()
    assert cfg.run.output_root == local["run.output_root"]
    assert cfg.submission.working_zip_path == local["submission.working_zip_path"]
    assert cfg.output.bundle_dir == local["output.bundle_dir"]
    # An explicit value is never rewritten.
    cfg = load_config(set_overrides=["run.output_root=/somewhere/else"])
    assert cfg.run.output_root == "/somewhere/else"


def test_kaggle_paths_are_kept_on_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLSFT_PLATFORM", "kaggle")
    cfg = load_config()
    assert cfg.run.output_root == "/kaggle/working/runs"
    raw = {"run": {"output_root": "/kaggle/working/runs"}}
    assert resolve_default_paths(raw, on_kaggle=True) == [] and raw["run"]["output_root"] == "/kaggle/working/runs"


def test_val_every_steps_triggers_validation() -> None:
    from pipeline.multi_lora_sft.training.callbacks import MultiLoraTrainingCallback

    wsd = SimpleNamespace(now=lambda: 0.0, on_step_end=lambda *a, **k: None, should_stop=lambda now: False,
                          step=1, t_start=0.0, deadline=1e9, projected_t_end=lambda now: 0.0)
    cb = MultiLoraTrainingCallback(
        adapter_name="main_lora", wsd_controller=wsd, metrics_jsonl_path=Path("unused.jsonl"),
        checkpoint_root=Path("unused"), metrics_tracker={}, sampler=None, dataset_total_tokens=1, samples_per_step=1,
        val_every_min=600.0, val_every_steps=4, has_eval_dataset=True, dead_check_step=10_000,
        throughput_project_step=10_000,
    )
    fired = []
    for step in range(1, 13):
        control = SimpleNamespace(should_evaluate=False, should_training_stop=False)
        state = SimpleNamespace(global_step=step, is_world_process_zero=True, epoch=0.0)
        cb.on_step_end(SimpleNamespace(world_size=1), state, control)
        if control.should_evaluate:
            fired.append(step)
    assert fired == [4, 8, 12]


def _ctx(tmp_path: Path, run_id: str = "insights"):
    cfg = load_config(set_overrides=[
        f"run.output_root={(tmp_path / 'runs').as_posix()}", f"run.run_id={run_id}",
        "adapters=[{name: main_lora, roles: [root]}]",
    ])
    ctx = RunContext.create(cfg, t0=1000.0)
    ctx.ensure_dirs()
    return cfg, ctx


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_report_insights_flag_overfitting_unused_budget_and_eval_problems(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path)
    _write_jsonl(ctx.metrics_dir / "main_lora.jsonl", [
        {"step": s, "train_loss": 0.3 - s * 0.005, "grad_norm": 0.5, "lr": 1e-4, "tokens_per_sec": 400.0} for s in range(1, 49)
    ])
    _write_jsonl(ctx.metrics_dir / "main_lora_val.jsonl", [
        {"step": 8, "eval_loss": 0.40}, {"step": 16, "eval_loss": 0.33}, {"step": 24, "eval_loss": 0.36},
    ])
    ctx.write_status(PhaseStatus(started="t", phase="train:main_lora", state="ok", planned_duration_sec=36000.0,
                                 actual_duration_sec=8700.0))
    ctx.write_status(PhaseStatus(started="t", phase="eval", state="ok"))
    ctx.eval_score_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.eval_score_path.write_text(json.dumps({"tasks": [
        {"task_id": "a", "result": "FAIL", "patch_applied": False, "turns": 12, "tool_call_errors": 2},
        {"task_id": "b", "result": "TIMEOUT", "patch_applied": False, "turns": 40, "tool_call_errors": 0},
        {"task_id": "c", "result": "PASS", "patch_applied": True, "turns": 9, "tool_call_errors": 0},
    ]}), encoding="utf-8")

    assert [v["step"] for v in val_rows(ctx, "main_lora")] == [8, 16, 24]
    m = eval_metrics(json.loads(ctx.eval_score_path.read_text(encoding="utf-8")))
    assert m == {"tasks": 3, "passed": 1, "timeouts": 1, "patch_applied": 1, "crashed": 0, "mean_turns": 20.3,
                 "tool_call_errors": 2}

    findings = "\n".join(f"{sev} {f} -> {step}" for sev, f, step in diagnosis(ctx, cfg, {}))
    assert "likely over-fitting" in findings
    assert "bottomed at step 16" in findings
    assert "2.4 h of a 10.0 h" in findings
    assert "Patches applied on only 1/3" in findings and "timed out" in findings and "tool-call error" in findings

    curve = "\n".join(learning_curve_lines(ctx, cfg))
    assert "| 16 | 0.3300 **(best)** |" in curve


def test_diagnosis_points_preflight_failures_at_the_validator(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path, "insights_pf")
    ctx.write_status(PhaseStatus(started="t", phase="preflight", state="failed",
                                 error="389 record(s) are missing the required 'agent' field and 0 matched."))
    rows = diagnosis(ctx, cfg, {})
    assert rows[0][0] == "FAIL" and "validate_dataset.py" in rows[0][2]


# ---- speed / monitoring options ----------------------------------------------------------------------------------


def test_flash_attention_falls_back_for_wide_heads(monkeypatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    from pipeline.multi_lora_sft.training.model import resolve_attn_implementation

    gemma4 = SimpleNamespace(head_dim=256, global_head_dim=512)
    narrow = SimpleNamespace(head_dim=128, global_head_dim=None)
    assert resolve_attn_implementation("sdpa", gemma4, True) == ("sdpa", None)
    assert resolve_attn_implementation("flash_attention_2", narrow, False)[0] == "sdpa"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "flash_attn" else object())
    impl, why = resolve_attn_implementation("flash_attention_2", narrow, True)
    assert impl == "sdpa" and "not installed" in why
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    impl, why = resolve_attn_implementation("flash_attention_2", gemma4, True)
    assert impl == "sdpa" and "512" in why
    assert resolve_attn_implementation("flash_attention_2", narrow, True) == ("flash_attention_2", None)


def test_prequantized_dir_detection(tmp_path: Path) -> None:
    import json

    from pipeline.multi_lora_sft.config import load_config
    from pipeline.multi_lora_sft.training.model import apply_liger_kernels, is_prequantized_nf4_dir

    assert not is_prequantized_nf4_dir(None)
    assert not is_prequantized_nf4_dir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "bitsandbytes",
                                                                            "load_in_4bit": True}}))
    assert not is_prequantized_nf4_dir(tmp_path)  # no weights yet
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert is_prequantized_nf4_dir(tmp_path)

    cfg = load_config(set_overrides=["model.use_liger_kernel=true", f"model.prequantized_path={tmp_path.as_posix()}"])
    assert cfg.model.use_liger_kernel and cfg.model.prequantized_path == tmp_path.as_posix()
    assert isinstance(apply_liger_kernels(object()), str)  # never raises, installed or not
