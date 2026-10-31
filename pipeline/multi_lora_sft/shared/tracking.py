"""Experiment tracking policy (Weights & Biases) that can never break or slow a run.

Rules, decided in one place for every backend:

* Optuna study running -> W&B is OFF (hundreds of tiny trials would flood the project and add startup latency).
* Only rank 0 logs.
* Kaggle competition sessions have no internet -> default is `offline` (files under the run dir, uploadable later with
  `wandb sync`). `online` needs a key AND a reachable network; otherwise it degrades to `offline` with a logged reason.
* Any W&B exception disables W&B for the rest of the run; training continues. The key is read from the environment
  (`WANDB_API_KEY`) only, never from a config file, so a public notebook cannot leak one.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODES = ("off", "offline", "online", "auto")


@dataclass
class TrackingDecision:
    mode: str  # "off" | "offline" | "online"
    reason: str


def internet_reachable(host: str = "api.wandb.ai", port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def decide_tracking(
    *,
    requested: str,
    tuning_active: bool,
    is_main: bool,
    has_key: bool,
    disable_during_tuning: bool = True,
    probe: Any = internet_reachable,
) -> TrackingDecision:
    if requested not in MODES:
        return TrackingDecision("off", f"unknown tracking mode {requested!r}")
    if tuning_active and disable_during_tuning:
        return TrackingDecision("off", "Optuna study active (tracking.disable_during_tuning)")
    if not is_main:
        return TrackingDecision("off", "not the main process")
    if requested == "off":
        return TrackingDecision("off", "tracking.wandb = off")
    if requested == "offline":
        return TrackingDecision("offline", "tracking.wandb = offline")
    if not has_key:
        return TrackingDecision("offline", f"tracking.wandb = {requested} but WANDB_API_KEY is not set")
    if requested == "auto" and not probe():
        return TrackingDecision("offline", "no internet; logging offline")
    if requested == "online" and not probe():
        return TrackingDecision("offline", "tracking.wandb = online but the network is unreachable")
    return TrackingDecision("online", f"tracking.wandb = {requested}")


class Tracker:
    """Thin W&B wrapper. `log` / `summary` / `finish` are safe to call in every mode."""

    def __init__(
        self,
        *,
        requested: str,
        tuning_active: bool,
        is_main: bool,
        run_dir: str | Path,
        project: str,
        run_name: str,
        group: str | None = None,
        config: dict[str, Any] | None = None,
        disable_during_tuning: bool = True,
    ) -> None:
        self.decision = decide_tracking(
            requested=requested,
            tuning_active=tuning_active,
            is_main=is_main,
            has_key=bool(os.environ.get("WANDB_API_KEY")),
            disable_during_tuning=disable_during_tuning,
        )
        self.mode = self.decision.mode
        self._run: Any = None
        self.failed: str | None = None
        logger.info("Tracking: %s (%s)", self.mode, self.decision.reason)
        if self.mode == "off":
            return
        for attempt in ([self.mode, "offline"] if self.mode == "online" else [self.mode]):
            try:
                import wandb

                wdir = Path(run_dir) / "wandb"
                wdir.mkdir(parents=True, exist_ok=True)
                os.environ["WANDB_MODE"] = attempt
                os.environ.setdefault("WANDB_SILENT", "true")
                os.environ.setdefault("WANDB_DIR", str(wdir))
                os.environ.setdefault("WANDB_CONSOLE", "off")
                self._run = wandb.init(
                    project=project, name=run_name, group=group, dir=str(wdir), config=config or {},
                    reinit=True, mode=attempt, settings=wandb.Settings(silent=True, console="off", init_timeout=120),
                )
                self.mode = attempt
                where = getattr(self._run, "url", None) if attempt == "online" else getattr(self._run, "dir", wdir)
                logger.info("W&B %s run: %s%s", attempt, where,
                            "" if attempt == "online" else "  (upload later: wandb sync <run_dir>/wandb/offline-run-*)")
                return
            except BaseException as exc:  # noqa: BLE001 - W&B must never take a training run down
                if attempt == "online":  # bad key / blocked network: keep the curves locally instead of losing them
                    logger.warning("W&B online start failed (%s: %s); falling back to offline", type(exc).__name__, exc)
                    self.decision = TrackingDecision("offline", f"online start failed: {type(exc).__name__}")
                    continue
                self._disable(exc)

    def _disable(self, exc: BaseException) -> None:
        self.failed = f"{type(exc).__name__}: {exc}"
        logger.warning("W&B disabled for this run (%s)", self.failed)
        self.mode, self._run = "off", None

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.log({k: v for k, v in metrics.items() if v is not None}, step=int(step))
        except BaseException as exc:  # noqa: BLE001
            self._disable(exc)

    def summary(self, values: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            for k, v in values.items():
                if isinstance(v, (int, float, str, bool)):
                    self._run.summary[k] = v
        except BaseException as exc:  # noqa: BLE001
            self._disable(exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except BaseException as exc:  # noqa: BLE001
            self._disable(exc)
        self._run = None
