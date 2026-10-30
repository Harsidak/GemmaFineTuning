"""Session orchestrator (`python -m pipeline.multi_lora_sft.run`).

Implements design_doc.md §5, §6.1, §6.2, and §10:
- Resolves and validates config + `--set` overrides; writes `resolved_config.yaml` and initial
  `manifest.json` before launching any phase.
- Runs each phase in an isolated subprocess with hard wall-clock deadlines and full process-tree
  termination on timeout.
- Implements OOM fallback retries (`train.oom_fallbacks`), mid-run crash restarts from latest
  checkpoint (`train.max_restarts`, `train.min_restart_min`), dynamic leftover time redistribution
  across adapters, and **guaranteed execution of `package` and `report`**.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clock import SessionClock
from .config import AdapterSpec, PipelineConfig, load_config
from .runctx import (
    PhaseStatus,
    RunContext,
    compute_pipeline_code_hash,
    configure_phase_logging,
    utc_iso,
)

CommandBuilder = Callable[[str, RunContext, float, dict[str, Any] | None], list[str]]

OOM_SIGNATURES: tuple[str, ...] = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "hip out of memory",
)


@dataclass
class OrchestratorResult:
    exit_code: int
    run_id: str
    run_dir: Path
    submission_zip_path: Path | None
    report_path: Path


def _terminate_process_tree(proc: subprocess.Popen[Any], grace_sec: float = 3.0) -> None:
    """Terminates `proc` and all child processes cleanly across POSIX and Windows."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        with contextlib.suppress(OSError):
            proc.kill()
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + grace_sec
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
            if proc.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            with contextlib.suppress(OSError):
                proc.kill()
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        proc.wait(timeout=5.0)


def default_phase_command_builder(
    cfg: PipelineConfig,
) -> CommandBuilder:
    """Builds the default `python -m pipeline.multi_lora_sft.phases.<phase>` subprocess argv."""

    def _build(
        phase: str,
        ctx: RunContext,
        deadline: float,
        extra: dict[str, Any] | None = None,
    ) -> list[str]:
        extra = extra or {}
        if phase.startswith("train:"):
            adapter_name = phase.split(":", 1)[1]
            eff = cfg.resolve_adapter(adapter_name)
            manifest = ctx.read_manifest()
            gpu_cnt = len(manifest.get("gpus") or [])
            n_proc = eff.train.num_processes if eff.train.num_processes is not None else gpu_cnt

            if eff.train.parallelism == "ddp" and n_proc and n_proc > 1:
                cmd = [
                    sys.executable,
                    "-m",
                    "accelerate.commands.launch",
                    "--num_processes",
                    str(n_proc),
                    "--mixed_precision",
                    "bf16",
                    "-m",
                    "pipeline.multi_lora_sft.phases.train",
                ]
            else:
                cmd = [sys.executable, "-m", "pipeline.multi_lora_sft.phases.train"]

            cmd.extend(["--run-dir", str(ctx.run_dir), "--adapter", adapter_name, "--deadline", f"{deadline:.3f}"])
            if extra.get("oom_override"):
                cmd.extend(["--oom-override-json", json.dumps(extra["oom_override"])])
            return cmd

        module_map = {
            "setup": "pipeline.multi_lora_sft.phases.setup",
            "preflight": "pipeline.multi_lora_sft.phases.preflight",
            "export": "pipeline.multi_lora_sft.phases.export",
            "package": "pipeline.multi_lora_sft.phases.package",
            "eval": "pipeline.multi_lora_sft.phases.evaluate",
            "report": "pipeline.multi_lora_sft.phases.report",
        }
        mod = module_map[phase]
        return [
            sys.executable,
            "-m",
            mod,
            "--run-dir",
            str(ctx.run_dir),
            "--deadline",
            f"{deadline:.3f}",
        ]

    return _build


def _detect_oom(log_file: Path, status_dict: dict[str, Any] | None) -> bool:
    if status_dict and status_dict.get("error"):
        err_low = str(status_dict["error"]).lower()
        if any(sig in err_low for sig in OOM_SIGNATURES):
            return True
    if log_file.is_file():
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-16384:].lower()
        if any(sig in tail for sig in OOM_SIGNATURES):
            return True
    return False


class Orchestrator:
    """Drives all 7 phases of a multi-LoRA SFT session under a wall-clock budget."""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        clock: SessionClock | None = None,
        ctx: RunContext | None = None,
        command_builder: CommandBuilder | None = None,
        poll_interval_sec: float = 0.25,
    ) -> None:
        self.cfg = cfg
        self.clock = clock or SessionClock(cfg)
        self.ctx = ctx or RunContext.create(cfg, self.clock.t0)
        self.command_builder = command_builder or default_phase_command_builder(cfg)
        self.poll_interval_sec = max(0.05, float(poll_interval_sec))
        self.logger = configure_phase_logging(self.ctx.log_path("orchestrator"))

    def _run_subprocess_phase(
        self,
        phase: str,
        deadline: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Launches `phase` in an isolated subprocess, enforcing `deadline` via process-group kill."""
        t_start = self.clock.now()
        cmd = self.command_builder(phase, self.ctx, deadline, extra)
        log_file = self.ctx.log_path(phase)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        repo_root = str(Path(__file__).resolve().parents[2])
        existing_pp = env.get("PYTHONPATH", "")
        if repo_root not in existing_pp.split(os.pathsep):
            env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root

        popen_kwargs: dict[str, Any] = {
            "cwd": repo_root,
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self.logger.info(
            "==> Phase %s | budget=%.1fs | deadline=%s",
            phase,
            max(0.0, deadline - t_start),
            utc_iso(deadline),
        )

        # Clear any prior attempt's status JSON (e.g. before OOM fallback or crash restart)
        # so `read_status(phase)` after exit only reflects this subprocess invocation.
        st_file = self.ctx.status_path(phase)
        if st_file.exists():
            st_file.unlink()

        timed_out = False
        with open(log_file, "a", encoding="utf-8") as log_fh:
            log_fh.write(f"\n--- [{utc_iso(t_start)}] LAUNCH {phase}: {' '.join(cmd)} ---\n")
            log_fh.flush()
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, **popen_kwargs)

            while proc.poll() is None:
                if self.clock.now() >= deadline:
                    timed_out = True
                    self.logger.warning("Phase %s reached hard deadline; terminating process tree", phase)
                    _terminate_process_tree(proc)
                    break
                time.sleep(self.poll_interval_sec)

            rc = proc.returncode if proc.returncode is not None else -9

        t_end = self.clock.now()
        existing_status = self.ctx.read_status(phase)

        if timed_out:
            status = PhaseStatus(
                phase=phase,
                state="timeout",
                started=utc_iso(t_start),
                ended=utc_iso(t_end),
                started_ts=t_start,
                ended_ts=t_end,
                planned_deadline=utc_iso(deadline),
                planned_deadline_ts=deadline,
                planned_duration_sec=round(deadline - t_start, 2),
                actual_duration_sec=round(t_end - t_start, 2),
                error=f"Killed at hard phase deadline ({utc_iso(deadline)})",
                outputs=(existing_status or {}).get("outputs", {}),
                notes=[*((existing_status or {}).get("notes") or []), "Process tree terminated on timeout"],
            )
            self.ctx.write_status(status)
            return status.to_dict()

        if existing_status is not None:
            # Ensure orchestrator wall-clock fields are populated if child wrote status
            if rc != 0 and existing_status.get("state") == "ok":
                existing_status["state"] = "failed"
                existing_status["error"] = f"Subprocess exited with code {rc} despite ok status"
                self.ctx.status_path(phase).write_text(
                    json.dumps(existing_status, indent=2), encoding="utf-8"
                )
            return existing_status

        state = "ok" if rc == 0 else "failed"
        status = PhaseStatus(
            phase=phase,
            state=state,
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline),
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2),
            actual_duration_sec=round(t_end - t_start, 2),
            error=None if rc == 0 else f"Subprocess exited with code {rc} without writing status JSON",
        )
        self.ctx.write_status(status)
        return status.to_dict()

    def _write_skipped_status(self, phase: str, reason: str) -> dict[str, Any]:
        now = self.clock.now()
        st = PhaseStatus(
            phase=phase,
            state="skipped",
            started=utc_iso(now),
            ended=utc_iso(now),
            started_ts=now,
            ended_ts=now,
            actual_duration_sec=0.0,
            notes=[reason],
        )
        self.ctx.write_status(st)
        return st.to_dict()

    def _run_adapter_with_retries(
        self,
        spec: AdapterSpec,
        deadline: float,
    ) -> dict[str, Any]:
        """Executes `train:<adapter>` with OOM fallback retries and crash restarts (§6.2)."""
        eff = self.cfg.resolve_adapter(spec)
        phase = f"train:{spec.name}"
        log_file = self.ctx.log_path(phase)

        oom_idx = 0
        restarts = 0
        current_oom_override: dict[str, Any] | None = None

        while True:
            st = self._run_subprocess_phase(
                phase,
                deadline,
                extra={"oom_override": current_oom_override} if current_oom_override else None,
            )
            if st.get("state") in {"ok", "timeout"}:
                return st

            rem_sec = deadline - self.clock.now()
            is_oom = _detect_oom(log_file, st)

            if is_oom and oom_idx < len(eff.train.oom_fallbacks) and rem_sec >= 5.0:
                current_oom_override = dict(eff.train.oom_fallbacks[oom_idx])
                oom_idx += 1
                self.logger.warning(
                    "OOM detected in %s; retrying with fallback #%d: %s",
                    phase,
                    oom_idx,
                    current_oom_override,
                )
                continue

            min_restart_sec = eff.train.min_restart_min * 60.0
            if not is_oom and restarts < eff.train.max_restarts and rem_sec >= min_restart_sec:
                restarts += 1
                self.logger.warning(
                    "Crash in %s; restarting from latest checkpoint (%d/%d, rem=%.1fs)",
                    phase,
                    restarts,
                    eff.train.max_restarts,
                    rem_sec,
                )
                continue

            return st

    def run(self) -> OrchestratorResult:
        """Executes `setup -> preflight -> train:* -> export -> package -> eval -> report`."""
        self.ctx.ensure_dirs()
        self.cfg.save_yaml(self.ctx.resolved_config_path)

        git_sha = self.cfg.run.git_sha or os.environ.get("GIT_SHA")
        self.ctx.update_manifest(
            {
                "run_id": self.ctx.run_id,
                "run_dir": str(self.ctx.run_dir),
                "mode": self.cfg.run.mode,
                "seed": self.cfg.run.seed,
                "session_t0": self.clock.t0,
                "session_t0_iso": utc_iso(self.clock.t0),
                "planned_train_deadline_iso": utc_iso(self.clock.plan.train_deadline),
                "planned_eval_deadline_iso": utc_iso(self.clock.plan.eval_deadline),
                "code_sha256": compute_pipeline_code_hash(),
                "git_sha": git_sha,
            }
        )

        setup_ok = False
        preflight_ok = False
        valid_adapter_names: set[str] = set()
        sup_tokens_map: dict[str, int] = {}

        try:
            # 1. Setup
            setup_st = self._run_subprocess_phase("setup", self.clock.phase_deadline("setup"))
            setup_ok = setup_st.get("state") == "ok"

            if not setup_ok:
                self._write_skipped_status("preflight", "Skipped because setup failed")
                for spec in self.cfg.adapters:
                    self._write_skipped_status(f"train:{spec.name}", "Skipped because setup failed")
                self._write_skipped_status("export", "Skipped because setup failed")
            else:
                # 2. Preflight
                pf_st = self._run_subprocess_phase("preflight", self.clock.phase_deadline("preflight"))
                preflight_ok = pf_st.get("state") == "ok"
                pf_out = pf_st.get("outputs") or {}
                valid_adapter_names = set(pf_out.get("valid_adapters") or [a.name for a in self.cfg.adapters])
                sup_tokens_map = {
                    k: int(v) for k, v in (pf_out.get("supervised_tokens_by_adapter") or {}).items()
                }

                if not preflight_ok:
                    for spec in self.cfg.adapters:
                        self._write_skipped_status(f"train:{spec.name}", "Skipped because preflight failed")
                    self._write_skipped_status("export", "Skipped because preflight failed")
                else:
                    # 3. Train each valid adapter sequentially, redistributing leftover time dynamically
                    trainable_specs = [a for a in self.cfg.adapters if a.name in valid_adapter_names]
                    for spec in self.cfg.adapters:
                        if spec.name not in valid_adapter_names:
                            self._write_skipped_status(
                                f"train:{spec.name}", "Skipped because adapter failed preflight"
                            )

                    for idx, spec in enumerate(trainable_specs):
                        remaining_specs = trainable_specs[idx:]
                        adapter_deadline = self.clock.next_adapter_deadline(
                            spec, remaining_specs, sup_tokens_map
                        )
                        if adapter_deadline - self.clock.now() <= 1.0:
                            self._write_skipped_status(
                                f"train:{spec.name}", "Skipped because training time budget is exhausted"
                            )
                            continue
                        self._run_adapter_with_retries(spec, adapter_deadline)

                    # 4. Export
                    self._run_subprocess_phase("export", self.clock.phase_deadline("export"))

        finally:
            # 5. Package — ALWAYS runs (§6.2 Goal G4: "Always ship")
            pkg_st = self._run_subprocess_phase("package", self.clock.phase_deadline("package"))
            pkg_ok = pkg_st.get("state") == "ok" and self.ctx.submission_zip_path.is_file()

            # 6. Eval — runs only if setup succeeded, package succeeded, and eval is enabled
            if not self.cfg.eval.enabled:
                self._write_skipped_status("eval", "eval.enabled is False")
            elif not setup_ok:
                self._write_skipped_status("eval", "Skipped because setup failed")
            elif not pkg_ok:
                self._write_skipped_status("eval", "Skipped because package failed")
            elif self.clock.remaining_until(self.clock.phase_deadline("eval")) <= 5.0:
                self._write_skipped_status("eval", "Skipped because eval deadline already passed")
            else:
                self._run_subprocess_phase("eval", self.clock.phase_deadline("eval"))

            # 7. Report — ALWAYS runs last
            self._run_subprocess_phase("report", self.clock.phase_deadline("report"))

        if self.ctx.report_path.is_file():
            self.logger.info("Run finished. Summary written to %s", self.ctx.report_path)

        # Exit code policy (§6.2):
        # - Non-zero if package failed (`2`) or setup / global preflight hard-failed (`1`).
        # - Zero if setup + preflight passed and a valid submission.zip was packaged (even if eval failed).
        if not pkg_ok:
            exit_code = 2
        elif not setup_ok or not preflight_ok:
            exit_code = 1
        else:
            exit_code = 0

        return OrchestratorResult(
            exit_code=exit_code,
            run_id=self.ctx.run_id,
            run_dir=self.ctx.run_dir,
            submission_zip_path=self.ctx.submission_zip_path if pkg_ok else None,
            report_path=self.ctx.report_path,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time-budgeted Multi-LoRA SFT Orchestrator for Gemma 4 SWE-bench"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g. pipeline/multi_lora_sft/configs/31b_full.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key using dotted notation with YAML-parsed value (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set_overrides)
    orch = Orchestrator(cfg)
    result = orch.run()
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
