"""Framework-neutral training guards: NaN/Inf, loss divergence, wall-clock timers and best-checkpoint selection.

The TRL `TrainerCallback` and the Tunix training loop both feed these the same numbers, so the policy
(how many bad logs abort, what counts as a spike, when to stop early) is defined exactly once.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class NonFiniteLossError(RuntimeError):
    """Training produced NaN/Inf loss or gradients for several consecutive logging windows."""


class LossDivergedError(RuntimeError):
    """The training loss exploded relative to its running average for several consecutive logs."""


class NonFiniteGuard:
    """Counts consecutive non-finite loss / grad-norm observations and raises after `limit`."""

    def __init__(self, limit: int = 3, label: str = "") -> None:
        self.limit = max(1, int(limit))
        self.label = label
        self.streak = 0

    def observe(self, loss: float | None, grad_norm: float | None = None, step: int | None = None) -> bool:
        bad = (loss is not None and not math.isfinite(float(loss))) or (
            grad_norm is not None and not math.isfinite(float(grad_norm))
        )
        self.streak = self.streak + 1 if bad else 0
        if bad:
            logger.warning(
                "%s Non-finite loss/grad_norm at step %s (loss=%s, grad_norm=%s) [%d/%d]",
                self.label, step, loss, grad_norm, self.streak, self.limit,
            )
        if self.streak >= self.limit:
            raise NonFiniteLossError(
                f"{self.label} {self.streak} consecutive non-finite loss/grad-norm logs at step {step}; aborting so the run "
                "restarts from the last good checkpoint. Lower train.lr, raise schedule.warmup_steps, or check data for "
                "empty supervision."
            )
        return bad


class NoGradientError(RuntimeError):
    """The LoRA weights received an exactly-zero gradient on several consecutive supervised steps."""


class ZeroGradGuard:
    """Raises after `limit` consecutive steps with supervised tokens but a gradient norm of exactly 0.

    LoRA B starts at zero but A's partner gradient (dL/dB) does not, so a healthy first step always has a positive
    norm. Exactly 0 means the adapter is not in the forward pass (e.g. the LoRA interception missed the method that
    runs the model): without this guard such a run would burn its whole session and export an untrained adapter."""

    def __init__(self, limit: int = 3, label: str = "") -> None:
        self.limit = max(1, int(limit))
        self.label = label
        self.streak = 0

    def observe(self, grad_norm: float | None, supervised: int | None = None, step: int | None = None) -> None:
        if grad_norm is None or not math.isfinite(float(grad_norm)) or supervised == 0:
            return  # unknown, non-finite (NonFiniteGuard's job) or nothing to learn from: no evidence either way
        self.streak = self.streak + 1 if float(grad_norm) == 0.0 else 0
        if self.streak >= self.limit:
            raise NoGradientError(
                f"{self.label} gradient norm was exactly 0 on {self.streak} consecutive supervised steps (step {step}): "
                "the LoRA adapter is not receiving gradients, training would not change it. Check the LoRA attachment."
            )


class LossSpikeGuard:
    """EMA-based divergence detector. `factor <= 0` disables it. Warm-up logs never count."""

    def __init__(self, factor: float = 0.0, limit: int = 5, warmup_logs: int = 20, beta: float = 0.95, label: str = "") -> None:
        self.factor = float(factor)
        self.limit = max(1, int(limit))
        self.warmup_logs = max(1, int(warmup_logs))
        self.beta = float(beta)
        self.label = label
        self.ema: float | None = None
        self.seen = 0
        self.streak = 0

    def observe(self, loss: float, step: int | None = None) -> bool:
        if self.factor <= 0.0 or not math.isfinite(loss):
            return False
        self.seen += 1
        spike = self.ema is not None and self.seen > self.warmup_logs and loss > self.factor * self.ema
        self.streak = self.streak + 1 if spike else 0
        if not spike:  # spikes must not drag the baseline up
            self.ema = loss if self.ema is None else self.beta * self.ema + (1.0 - self.beta) * loss
        else:
            logger.warning("%s Loss spike at step %s: %.4f vs EMA %.4f [%d/%d]", self.label, step, loss, self.ema, self.streak, self.limit)
        if self.streak >= self.limit:
            raise LossDivergedError(
                f"{self.label} loss {loss:.4f} exceeded {self.factor}x its running average {self.ema:.4f} for "
                f"{self.streak} consecutive logs at step {step}; aborting to the last good checkpoint."
            )
        return spike

    def state_dict(self) -> dict[str, Any]:
        return {"ema": self.ema, "seen": self.seen}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema = state.get("ema")
        self.seen = int(state.get("seen", 0))


class IntervalTimer:
    """`due(now)` is true once `every_sec` has elapsed since the last `mark`."""

    def __init__(self, every_sec: float, start: float) -> None:
        self.every_sec = max(1.0, float(every_sec))
        self.last = float(start)

    def due(self, now: float) -> bool:
        return (now - self.last) >= self.every_sec

    def mark(self, now: float) -> None:
        self.last = float(now)


@dataclass
class BestDecision:
    is_best: bool
    should_stop: bool
    best_step: int | None
    best_loss: float | None
    evals_without_improvement: int


class BestTracker:
    """Tracks the lowest validation loss; optional early stopping after `patience` evals without `min_delta` gain."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0) -> None:
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.best_loss: float | None = None
        self.best_step: int | None = None
        self.stale = 0
        self.history: list[dict[str, float]] = []

    def update(self, step: int, val_loss: float) -> BestDecision:
        if not math.isfinite(val_loss):  # a NaN eval is never "best" and never counts as progress
            self.stale += 1
            return BestDecision(False, self.patience > 0 and self.stale >= self.patience, self.best_step, self.best_loss, self.stale)
        self.history.append({"step": int(step), "val_loss": float(val_loss)})
        improved = self.best_loss is None or val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_step, self.stale = float(val_loss), int(step), 0
        else:
            self.stale += 1
        return BestDecision(improved, self.patience > 0 and self.stale >= self.patience, self.best_step, self.best_loss, self.stale)

    def state_dict(self) -> dict[str, Any]:
        return {"best_loss": self.best_loss, "best_step": self.best_step, "stale": self.stale, "history": self.history[-200:]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_loss = state.get("best_loss")
        self.best_step = state.get("best_step")
        self.stale = int(state.get("stale", 0))
        self.history = list(state.get("history", []))
