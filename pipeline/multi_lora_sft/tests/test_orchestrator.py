"""Unit tests for `pipeline.multi_lora_sft.run.Orchestrator` fault tolerance & timeouts (§6.2, §11)."""

from __future__ import annotations

import json
import logging
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


def test_subprocess_phase_log_lines_not_duplicated_and_fsdp_launch_flags(
    tmp_path: Path, repo_root: Path
) -> None:
    from pipeline.multi_lora_sft.run import phase_subprocess_env
    from pipeline.multi_lora_sft.runctx import configure_phase_logging

    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        repo_root / "pipeline" / "multi_lora_sft" / "configs" / "31b_full.yaml",
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=orch_dedup_logs",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
            "eval.enabled=false",
        ],
    )
    env = phase_subprocess_env(cfg, str(repo_root))
    assert env["USE_TF"] == "0"

    ctx = RunContext.create(cfg, 1_000_000.0)
    ctx.update_manifest({"gpus": [{"index": i, "name": "L4"} for i in range(4)]})
    builder = default_phase_command_builder(cfg)
    train_cmd = builder("train:main_lora", ctx, 1_001_000.0, None)
    assert "--multi_gpu" in train_cmd
    assert "--num_processes" in train_cmd and train_cmd[train_cmd.index("--num_processes") + 1] == "4"

    # Subprocess with stdout redirected to setup.log must emit each logged line once, not twice.
    def _single_phase_builder(phase: str, c: RunContext, deadline: float, extra: dict[str, Any] | None = None) -> list[str]:
        if phase == "setup":
            code = (
                "from pathlib import Path; "
                "from pipeline.multi_lora_sft.runctx import configure_phase_logging; "
                f"lg = configure_phase_logging(Path({c.log_path('setup').as_posix()!r})); "
                "lg.info('UNIQUE_SETUP_' + 'MARKER_42')"
            )
            return [sys.executable, "-c", code]
        return [sys.executable, "-c", "pass"]

    orch = Orchestrator(cfg, ctx=ctx, command_builder=_single_phase_builder, poll_interval_sec=0.05)
    orch._run_subprocess_phase("setup", orch.clock.now() + 30.0)
    setup_log_text = ctx.log_path("setup").read_text(encoding="utf-8")
    assert setup_log_text.count("UNIQUE_SETUP_MARKER_42") == 1

    # Direct in-process call (where stdout is NOT the log file) must still write to the log file.
    inproc_log = tmp_path / "inproc.log"
    lg = configure_phase_logging(inproc_log)
    lg.info("INPROC_MARKER_99")
    for h in logging.getLogger().handlers:
        h.flush()
    assert inproc_log.read_text(encoding="utf-8").count("INPROC_MARKER_99") == 1


def test_preflight_failure_skips_eval_instead_of_serving_the_bare_model(tmp_path: Path, repo_root: Path) -> None:
    """2026-11-03: preflight rejected every record, yet eval spent ~6 min starting vLLM to score 0 tasks."""
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=orch_preflight_fail",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
            "output.bundle_dir=null",
        ]
    )
    real_builder = default_phase_command_builder(cfg)
    launched: list[str] = []

    def _cmd_builder(phase: str, ctx: RunContext, deadline: float, extra: dict[str, Any] | None = None) -> list[str]:
        launched.append(phase)
        if phase == "setup":
            return [sys.executable, "-c", "print('ok')"]
        if phase == "preflight":
            return [sys.executable, "-c", "import sys; sys.exit(1)"]
        return real_builder(phase, ctx, deadline, extra)

    orch = Orchestrator(cfg, command_builder=_cmd_builder, poll_interval_sec=0.05)
    res = orch.run()
    assert res.exit_code == 1
    assert "eval" not in launched
    statuses = {s["phase"]: s for s in orch.ctx.read_all_statuses()}
    assert statuses["eval"]["state"] == "skipped"
    assert "No trained adapter" in statuses["eval"]["notes"][0]
    log = orch.ctx.log_path("orchestrator").read_text(encoding="utf-8")
    assert "RUN FAILED before training: preflight did not pass" in log


def test_checkpoint_cleanup_only_deletes_what_the_bundle_holds(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=orch_cleanup",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"output.bundle_dir={bundle.as_posix()}",
            "adapters=[{name: main_lora, roles: [root]}, {name: tool_lora, roles: [code_analyzer]}]",
        ]
    )
    orch = Orchestrator(cfg, poll_interval_sec=0.05)
    orch.ctx.ensure_dirs()

    def make_ckpt(root: Path, step: int) -> Path:
        d = root / f"step_{step:06d}"
        d.mkdir(parents=True)
        (d / "optimizer.pt").write_bytes(b"x" * 4096)
        (d / "trainer_resume.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        (d / "adapter").mkdir()
        (d / "adapter" / "adapter_model.safetensors").write_bytes(b"y" * 16)
        return d

    main_root = orch.ctx.checkpoints_dir / "main_lora"
    tool_root = orch.ctx.checkpoints_dir / "tool_lora"
    for st in (10, 20):
        make_ckpt(main_root, st)
    newest_tool = make_ckpt(tool_root, 5)
    (bundle / "main_lora" / "step_000020").mkdir(parents=True)  # bundle holds main_lora's newest; nothing for tool_lora

    orch._free_disk_before_eval()
    assert not list(main_root.glob("step_*"))
    assert newest_tool.is_dir()
