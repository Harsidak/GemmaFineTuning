"""Time-aware Warmup-Stable-Decay (WSD) learning-rate schedule with checkpoint resume.

Implements design_doc.md §7.6 and §7.8:
- Warmup: linear warmup over `warmup_steps` optimizer steps.
- Stable: hold at peak `lr` (multiplier 1.0).
- Decay: `t_end = min(deadline, projected_epochs_end)` updated from EMA step duration;
  decay starts at `t_end - decay_frac * train_duration` and freezes `t_end` once started,
  falling linearly to `lr * final_lr_ratio`.
- Stop: triggers clean stop when `now >= t_end` or `step >= total_epoch_steps`.
- Resume (§7.8):
  - If decay had NOT started: resumes in stable (or warmup) phase under the new budget.
  - If decay HAD started: finishes decay over at most 10% of the new budget; then stops unless
    `resume_extend=True`, in which case it re-enters stable and decays again near the new `t_end`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class WSDScheduleController:
    """Stateful time-aware WSD controller backing `torch.optim.lr_scheduler.LambdaLR`."""

    def __init__(
        self,
        *,
        warmup_steps: int = 20,
        decay_frac: float = 0.15,
        final_lr_ratio: float = 0.1,
        ema_beta: float = 0.9,
        total_epoch_steps: int,
        t_start: float,
        deadline: float,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay_frac = float(decay_frac)
        self.final_lr_ratio = float(final_lr_ratio)
        self.ema_beta = float(ema_beta)
        self.total_epoch_steps = max(1, int(total_epoch_steps))
        self.t_start = float(t_start)
        self.deadline = float(deadline)
        self._now_fn = now_fn

        self.step: int = 0
        self.ema_step_sec: float | None = None
        self._last_step_ts: float = self.t_start
        self.prior_elapsed_sec: float = 0.0

        self.decay_started: bool = False
        self.ever_decayed: bool = False
        self.decay_start_time: float | None = None
        self.decay_start_ratio: float = 1.0
        self.frozen_t_end: float | None = None

        # Post-checkpoint resume mid-decay state (§7.8)
        self._in_resume_quick_decay: bool = False
        self._resume_quick_end: float | None = None
        self._resume_extend: bool = False

        self.last_multiplier: float = self._compute_multiplier(0, self.t_start)

    def now(self) -> float:
        return float(self._now_fn())

    def total_elapsed_sec(self, now: float | None = None) -> float:
        cur = self.now() if now is None else float(now)
        return self.prior_elapsed_sec + max(0.0, cur - self.t_start)

    def projected_t_end(self, now: float | None = None) -> float:
        cur = self.now() if now is None else float(now)
        if self.decay_started and self.frozen_t_end is not None:
            return self.frozen_t_end
        if self.ema_step_sec is not None and self.step < self.total_epoch_steps:
            rem_steps = max(0, self.total_epoch_steps - self.step)
            epoch_end = cur + rem_steps * self.ema_step_sec
            return min(self.deadline, epoch_end)
        return self.deadline

    def on_step_end(self, completed_step: int, now: float | None = None) -> None:
        """Records wall-clock duration of `completed_step` (1-indexed) and updates decay state."""
        cur = self.now() if now is None else float(now)
        dt = max(1e-6, cur - self._last_step_ts)
        self._last_step_ts = cur
        self.step = int(completed_step)

        if self.ema_step_sec is None:
            self.ema_step_sec = dt
        else:
            self.ema_step_sec = self.ema_beta * self.ema_step_sec + (1.0 - self.ema_beta) * dt

        self._maybe_transition_phase(cur)

    def _maybe_transition_phase(self, cur: float) -> None:
        # Handle completion of a resumed quick-decay window (§7.8)
        if self._in_resume_quick_decay and self._resume_quick_end is not None and cur >= self._resume_quick_end:
            if self._resume_extend:
                # Re-enter stable phase for the remainder of the new budget
                self._in_resume_quick_decay = False
                self._resume_quick_end = None
                self.decay_started = False
                self.decay_start_time = None
                self.decay_start_ratio = 1.0
                self.frozen_t_end = None
            else:
                return

        if not self.decay_started and self.decay_frac > 0.0:
            t_end = self.projected_t_end(cur)
            # Total duration spans prior_elapsed + current segment up to t_end
            total_duration = max(1e-6, self.prior_elapsed_sec + max(0.0, t_end - self.t_start))
            # Ensure decay window fits within current segment
            decay_window = min(self.decay_frac * total_duration, max(1e-6, t_end - self.t_start))
            t_decay = t_end - decay_window
            if cur >= t_decay:
                self.decay_started = True
                self.ever_decayed = True
                self.decay_start_time = t_decay
                self.decay_start_ratio = 1.0
                self.frozen_t_end = t_end

    def _compute_multiplier(self, step: int, cur: float) -> float:
        self._maybe_transition_phase(cur)

        # Resume quick-decay takes precedence if active
        if self._in_resume_quick_decay and self.decay_start_time is not None and self.frozen_t_end is not None:
            span = max(1e-6, self.frozen_t_end - self.decay_start_time)
            prog = min(1.0, max(0.0, (cur - self.decay_start_time) / span))
            return self.decay_start_ratio - prog * (self.decay_start_ratio - self.final_lr_ratio)

        # Warmup phase (unless decay must already start because time budget is tiny)
        warmup_mult = 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_mult = float(step + 1) / float(self.warmup_steps)

        if not self.decay_started or self.decay_start_time is None or self.frozen_t_end is None:
            return warmup_mult

        span = max(1e-6, self.frozen_t_end - self.decay_start_time)
        prog = min(1.0, max(0.0, (cur - self.decay_start_time) / span))
        decay_mult = self.decay_start_ratio - prog * (self.decay_start_ratio - self.final_lr_ratio)
        return min(warmup_mult, decay_mult)

    def lr_lambda(self, step: int) -> float:
        """Callable passed directly to `torch.optim.lr_scheduler.LambdaLR`."""
        cur = self.now()
        self.step = max(self.step, int(step))
        mult = self._compute_multiplier(int(step), cur)
        self.last_multiplier = mult
        return mult

    def should_stop(self, now: float | None = None) -> bool:
        """Returns True when either max epochs or the active/frozen time deadline is reached."""
        cur = self.now() if now is None else float(now)
        self._maybe_transition_phase(cur)
        if self.step >= self.total_epoch_steps:
            return True
        if (
            self._in_resume_quick_decay
            and not self._resume_extend
            and self.frozen_t_end is not None
            and cur >= self.frozen_t_end
        ):
            return True
        target_end = self.frozen_t_end if (self.decay_started and self.frozen_t_end is not None) else self.deadline
        return cur >= target_end

    def state_dict(self, now: float | None = None) -> dict[str, Any]:
        cur = self.now() if now is None else float(now)
        return {
            "step": self.step,
            "ema_step_sec": self.ema_step_sec,
            "prior_elapsed_sec": self.total_elapsed_sec(cur),
            "decay_started": self.decay_started,
            "ever_decayed": self.ever_decayed,
            "last_multiplier": self.last_multiplier,
            "frozen_t_end_offset": (
                (self.frozen_t_end - cur) if (self.decay_started and self.frozen_t_end is not None) else None
            ),
        }

    def load_resume_state(
        self,
        saved: dict[str, Any],
        *,
        new_t_start: float,
        new_deadline: float,
        extend: bool = False,
    ) -> None:
        """Restores schedule state for a resumed session under `[new_t_start, new_deadline]` (§7.8)."""
        self.t_start = float(new_t_start)
        self.deadline = float(new_deadline)
        self._last_step_ts = self.t_start
        self.step = int(saved.get("step", 0))
        self.ema_step_sec = saved.get("ema_step_sec")
        self.prior_elapsed_sec = float(saved.get("prior_elapsed_sec", 0.0))
        self.ever_decayed = bool(saved.get("ever_decayed", False))
        saved_decay_started = bool(saved.get("decay_started", False))
        saved_mult = float(saved.get("last_multiplier", 1.0))

        new_budget = max(1e-6, self.deadline - self.t_start)
        if not saved_decay_started:
            # Continue stable (or warmup) phase
            self.decay_started = False
            self.decay_start_time = None
            self.decay_start_ratio = 1.0
            self.frozen_t_end = None
            self._in_resume_quick_decay = False
            self._resume_quick_end = None
            self._resume_extend = extend
        else:
            # Finish decay quickly over at most 10% of the new budget (§7.8)
            quick_duration = 0.10 * new_budget
            saved_rem = saved.get("frozen_t_end_offset")
            if saved_rem is not None and float(saved_rem) > 0:
                quick_duration = min(quick_duration, float(saved_rem))
            quick_end = self.t_start + max(1e-6, quick_duration)

            self.decay_started = True
            self.ever_decayed = True
            self.decay_start_time = self.t_start
            self.decay_start_ratio = max(self.final_lr_ratio, saved_mult)
            self.frozen_t_end = quick_end
            self._in_resume_quick_decay = True
            self._resume_quick_end = quick_end
            self._resume_extend = bool(extend)

        self.last_multiplier = self._compute_multiplier(self.step, self.t_start)


def create_wsd_scheduler(optimizer: Any, controller: WSDScheduleController) -> Any:
    """Builds a PyTorch `LambdaLR` bound to `controller`."""
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer, lr_lambda=controller.lr_lambda)
