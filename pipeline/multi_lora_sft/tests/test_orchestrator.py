"""Unit tests for `pipeline.multi_lora_sft.run.Orchestrator` fault tolerance & timeouts (§6.2, §11)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pipeline.multi_lora_sft.clock import SessionClock
from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.run import Orchestrator, default_phase_command_builder
from pipeline.multi_lora_sft.runctx import RunContext


def test_orchestrator_setup_failure_still_packages_and_reports(tmp_path: Path, repo_root: Path) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=orch_setup_fail",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
            "eval.enabled=false",
        ]
    )
    real_builder = default_phase_command_builder(cfg)

    def _cmd_builder(phase: str, ctx: RunContext, deadline: float, extra: dict[str, Any] | None = None) -> list[str]:
        if phase == "setup":
            return [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"]
        return real_builder(phase, ctx, deadline, extra)

    orch = Orchestrator(cfg, command_builder=_cmd_builder, poll_interval_sec=0.05)
    res = orch.run()

    # Exits non-zero because setup failed, YET package and report still ran and built submission.zip!
    assert res.exit_code == 1
    assert res.submission_zip_path is not None and res.submission_zip_path.is_file()
    assert res.report_path.is_file()

    statuses = {s["phase"]: s["state"] for s in orch.ctx.read_all_statuses()}
    assert statuses["setup"] == "failed"
    assert statuses["preflight"] == "skipped"
    assert statuses["train:main_lora"] == "skipped"
    assert statuses["package"] == "ok"
    assert statuses["report"] == "ok"


def test_orchestrator_kills_hanging_phase_at_deadline_and_retries_oom(
    tmp_path: Path, repo_root: Path
) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=orch_hang_and_oom",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
            "eval.enabled=false",
            "train.oom_fallbacks=[{max_seq_length: 8192}]",
        ]
    )
    real_builder = default_phase_command_builder(cfg)
    oom_marker = tmp_path / "main_lora_first_attempt.flag"

    def _cmd_builder(phase: str, ctx: RunContext, deadline: float, extra: dict[str, Any] | None = None) -> list[str]:
        if phase in {"setup", "preflight"}:
            return [sys.executable, "-c", "print('ok')"]
        if phase == "train:main_lora":
            # First attempt simulates CUDA OOM; second attempt has oom_override and succeeds!
            if not oom_marker.exists():
                oom_marker.write_text("1", encoding="utf-8")
                return [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('RuntimeError: CUDA error: out of memory\\n'); sys.exit(1)",
                ]
            assert extra and extra.get("oom_override") == {"max_seq_length": 8192}
            return [sys.executable, "-c", "print('recovered from OOM')"]
        if phase == "train:tool_lora":
            # Hang well past its tight deadline -> orchestrator must kill process tree!
            return [sys.executable, "-c", "import time; time.sleep(60)"]
        if phase == "export":
            return [sys.executable, "-c", "print('export skipped fake')"]
        return real_builder(phase, ctx, deadline, extra)

    class ShortTrainClock(SessionClock):
        def next_adapter_deadline(self, current_adapter: Any, remaining_adapters: Any, supervised_tokens_by_adapter: Any = None) -> float:
            if current_adapter.name == "tool_lora":
                return self.now() + 1.5  # > 1.0s minimum launch window; kill hanging subprocess after 1.5s
            return self.now() + 15.0

    clock = ShortTrainClock(cfg)
    orch = Orchestrator(cfg, clock=clock, command_builder=_cmd_builder, poll_interval_sec=0.05)
    res = orch.run()

    assert res.exit_code == 0
    assert res.submission_zip_path is not None and res.submission_zip_path.is_file()
    assert res.report_path.is_file()

    statuses = {s["phase"]: s["state"] for s in orch.ctx.read_all_statuses()}
    assert statuses["train:main_lora"] == "ok"
    assert statuses["train:tool_lora"] == "timeout"
    assert statuses["package"] == "ok"
    assert statuses["report"] == "ok"

    manifest = json.loads(orch.ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["submission_zip_sha256"]
    assert any(p["phase"] == "train:tool_lora" and p["state"] == "timeout" for p in manifest["phases"])
