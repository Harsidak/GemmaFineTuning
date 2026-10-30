"""End-to-end CPU calibration test (`design_doc.md` §11).

Runs every phase (`setup -> preflight -> train:main_lora -> train:tool_lora -> export -> package -> report`)
in real subprocesses on CPU using the tiny Gemma-4-config model, `data/train_trajectories.jsonl`,
and `build/gemma-4-developer-agent/tasks.jsonl`, verifying that a valid `submission.zip` and `REPORT.md`
are produced.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.package import validate_staged_submission
from pipeline.multi_lora_sft.run import Orchestrator


def test_end_to_end_cpu_calibration_produces_valid_zip(
    tmp_path: Path, repo_root: Path, tiny_gemma4_dir: Path
) -> None:
    cfg_path = repo_root / "pipeline" / "multi_lora_sft" / "configs" / "e4b_calibration.yaml"
    train_jsonl = repo_root / "data" / "train_trajectories.jsonl"
    tasks_jsonl = repo_root / "build" / "gemma-4-developer-agent" / "tasks.jsonl"
    work_zip = tmp_path / "working" / "submission.zip"
    work_zip.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(
        cfg_path,
        set_overrides=[
            "run.mode=calibration",
            "run.calibration_minutes=10",
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=cpu_e2e_calibration",
            f"model.train_base_path={tiny_gemma4_dir.as_posix()}",
            f"model.served_path={tiny_gemma4_dir.as_posix()}",
            "model.quantization=fp32",
            f"data.train_files=[{train_jsonl.as_posix()}]",
            f"data.tasks_file={tasks_jsonl.as_posix()}",
            "train.parallelism=single",
            "train.grad_accum=1",
            "train.max_steps=2",
            "train.max_seq_length=8192",
            "parity.enabled=true",
            "parity.num_prompts=1",
            "parity.gen_tokens=4",
            f"submission.working_zip_path={work_zip.as_posix()}",
            "eval.enabled=false",
        ],
    )

    orch = Orchestrator(cfg, poll_interval_sec=0.1)
    result = orch.run()

    assert result.exit_code == 0
    assert result.submission_zip_path is not None and result.submission_zip_path.is_file()
    assert work_zip.is_file()
    assert result.report_path.is_file()

    statuses = {s["phase"]: s["state"] for s in orch.ctx.read_all_statuses()}
    assert statuses["setup"] == "ok"
    assert statuses["preflight"] == "ok"
    assert statuses["train:main_lora"] == "ok"
    assert statuses["train:tool_lora"] == "ok"
    assert statuses["export"] == "ok"
    assert statuses["package"] == "ok"
    assert statuses["eval"] == "skipped"
    assert statuses["report"] == "ok"

    # Both HF parity reference files were generated at the end of training
    assert (orch.ctx.parity_dir / "main_lora_hf.json").is_file()
    assert (orch.ctx.parity_dir / "tool_lora_hf.json").is_file()

    # Validate staged submission and zip contents
    val_info = validate_staged_submission(orch.ctx.submission_dir, expected_served_model="gemma-4-e4b-it-qat-w4a16-ct")
    assert sorted(val_info["present_adapters"]) == ["main_lora", "tool_lora"]

    with zipfile.ZipFile(result.submission_zip_path) as zf:
        names = set(zf.namelist())
        assert "agent.yaml" in names
        assert "sub_agents/code_analyzer.yaml" in names
        assert "adapters/main_lora/adapter_model.safetensors" in names
        assert "adapters/main_lora/adapter_config.json" in names
        assert "adapters/tool_lora/adapter_model.safetensors" in names
        assert "adapters/tool_lora/adapter_config.json" in names

    manifest = json.loads(orch.ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["leakage_passed"] is True
    assert manifest["adapter_sources"] == {"main_lora": "exported:main_lora", "tool_lora": "exported:tool_lora"}
    assert "Multi-LoRA SFT Run Report" in result.report_path.read_text(encoding="utf-8")
