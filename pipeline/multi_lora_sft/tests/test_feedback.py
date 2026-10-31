"""Run feedback: zero-gradient guard, Optuna trial clocks, system monitor, orchestrator heartbeat, report health section."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from pipeline.multi_lora_sft.clock import SessionClock
from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.run import Orchestrator, default_phase_command_builder
from pipeline.multi_lora_sft.runctx import RunContext
from pipeline.multi_lora_sft.shared.guards import NoGradientError, ZeroGradGuard
from pipeline.multi_lora_sft.shared.sysmon import (
    SystemSampler,
    format_system,
    read_latest,
    summarize,
    tail_line,
    wandb_system_metrics,
)
from pipeline.multi_lora_sft.shared.tuning import build_trial_sets


def test_zero_grad_guard_aborts_only_on_consecutive_supervised_zero_steps() -> None:
    g = ZeroGradGuard(3)
    g.observe(0.0, supervised=10, step=1)
    g.observe(0.0, supervised=10, step=2)
    g.observe(0.4, supervised=10, step=3)  # a real gradient resets the streak
    g.observe(0.0, supervised=0, step=4)  # nothing supervised: no evidence
    g.observe(None, supervised=10, step=5)  # unknown: no evidence
    g.observe(float("nan"), supervised=10, step=6)  # NonFiniteGuard's job
    g.observe(0.0, supervised=10, step=7)
    g.observe(0.0, supervised=None, step=8)
    with pytest.raises(NoGradientError, match="exactly 0"):
        g.observe(0.0, supervised=5, step=9)


def test_optuna_trials_get_their_own_session_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The notebook exports SESSION_T0 at its first cell; a trial inheriting it would see its window already used up."""
    monkeypatch.setenv("SESSION_T0", str(time.time() - 3 * 3600))  # notebook started 3 h ago
    cfg = load_config(set_overrides=["tuning.trial_minutes=10"])
    sets = build_trial_sets(cfg, "main_lora", [], study_dir=tmp_path, trial_no=3, shared_cache=None)
    trial_cfg = load_config(set_overrides=sets)
    clock = SessionClock(trial_cfg)
    assert abs(clock.t0 - time.time()) < 60
    assert clock.remaining_train_sec() > 60  # the trial really has its minutes


def test_system_monitor_helpers(tmp_path: Path) -> None:
    row = SystemSampler(disk_path=tmp_path).sample()  # must not raise on any OS, with or without psutil / GPUs / TPUs
    assert "ts" in row and "disk_free_gib" in row

    p = tmp_path / "system.jsonl"
    rows = [
        {"ts": time.time() - 60, "cpu_pct": 10.0, "ram_used_gib": 5.0, "ram_total_gib": 64.0, "disk_free_gib": 50.0,
         "tpu_duty_mean": 80.0, "tpu_hbm_max_gib": 12.0, "tpu_hbm_total_gib": 15.75, "tpu0_hbm_gib": 12.0},
        {"ts": time.time(), "cpu_pct": 30.0, "ram_used_gib": 9.0, "ram_total_gib": 64.0, "disk_free_gib": 40.0,
         "tpu_duty_mean": 90.0, "tpu_hbm_max_gib": 13.0, "tpu_hbm_total_gib": 15.75, "tpu0_hbm_gib": 13.0},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert json.loads(tail_line(p))["cpu_pct"] == 30.0
    assert read_latest(p)["tpu_duty_mean"] == 90.0
    assert read_latest(p, max_age_sec=None)["cpu_pct"] == 30.0
    assert read_latest(tmp_path / "missing.jsonl") == {}
    stale = tmp_path / "stale.jsonl"
    stale.write_text(json.dumps({"ts": time.time() - 3600, "cpu_pct": 1.0}) + "\n", encoding="utf-8")
    assert read_latest(stale, max_age_sec=300) == {}

    w = wandb_system_metrics(read_latest(p))
    assert w["sys/tpu_duty_mean"] == 90.0 and "sys/tpu0_hbm_gib" not in w  # per-device keys stay in the jsonl
    line = format_system(read_latest(p))
    assert "TPU duty 90%" in line and "HBM 13.0/15.8 GiB" in line and "disk 40 GiB free" in line
    s = summarize(p)
    assert s["samples"] == 2 and s["tpu_hbm_max_gib_max"] == 13.0 and s["disk_free_gib_min"] == 40.0
    assert s["tpu_duty_mean_avg"] == 85.0


def test_orchestrator_heartbeat_system_samples_and_health_report(tmp_path: Path, repo_root: Path) -> None:
    sample_tpl = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            "run.run_id=feedback",
            "model.served_model_id=gemma-4-e4b-it-qat-w4a16-ct",
            f"submission.template_dir={sample_tpl.as_posix()}",
            "submission.working_zip_path=null",
            "eval.enabled=false",
            "tracking.heartbeat_sec=0.3",
            "tracking.system_every_sec=0.2",
        ]
    )
    real_builder = default_phase_command_builder(cfg)

    def _cmd_builder(phase: str, ctx: RunContext, deadline: float, extra: dict[str, Any] | None = None) -> list[str]:
        if phase in {"setup", "preflight", "export"}:
            return [sys.executable, "-c", "print('ok')"]
        if phase.startswith("train:"):
            adapter = phase.split(":", 1)[1]
            rows = [
                {"step": 1, "epoch": 0.1, "train_loss": 2.5, "grad_norm": 1.2, "lr": 1e-4, "tokens_per_sec": 900.0, "peak_gpu_mem_mb": 1000.0},
                {"step": 7, "epoch": 0.7, "train_loss": 1.9, "grad_norm": 0.8, "lr": 1e-4, "tokens_per_sec": 1000.0, "peak_gpu_mem_mb": 1200.0},
            ]
            path = (ctx.metrics_dir / f"{adapter}.jsonl").as_posix()
            code = (
                "import json, time\n"
                f"with open({path!r}, 'w') as fh:\n"
                f"    fh.write(''.join(json.dumps(r) + '\\n' for r in {rows!r}))\n"
                "time.sleep(1.5)\n"
            )
            return [sys.executable, "-c", code]
        return real_builder(phase, ctx, deadline, extra)

    orch = Orchestrator(cfg, command_builder=_cmd_builder, poll_interval_sec=0.05)
    res = orch.run()
    assert res.report_path.is_file()

    log = orch.ctx.log_path("orchestrator").read_text(encoding="utf-8")
    beats = [line for line in log.splitlines() if "[heartbeat] train:" in line]
    assert beats, log[-2000:]
    assert any("step 7" in b and "loss 1.9000" in b and "grad_norm 0.8" in b for b in beats), beats
    assert any("session left" in b for b in beats)

    sys_rows = [json.loads(line) for line in (orch.ctx.metrics_dir / "system.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sys_rows) >= 2 and all("disk_free_gib" in r for r in sys_rows)

    report = res.report_path.read_text(encoding="utf-8")
    assert "## 5. Training Health & Resources" in report
    assert "2.5000 → 1.9000 ↓" in report
    assert "**Resources**" in report
