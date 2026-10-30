"""Session wall-clock planner, phase deadlines, and dynamic adapter time splitting.

Implements design_doc.md §5 and §7.7:
- All times are measured from session start `T0`.
- `train_deadline = T0 + session - margin - eval - package - export` (10h25m on 12h defaults).
- In `calibration` mode, total session time becomes `run.calibration_minutes` and phase budgets
  are proportionally scaled when the 12h defaults exceed the calibration window.
- Adapter budgets split the remaining training window by `time_share` or `supervised_tokens * max_epochs`,
  automatically redistributing any early-finish / skipped-adapter leftover time to subsequent adapters.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import AdapterSpec, PipelineConfig


@dataclass(frozen=True)
class PhasePlan:
    """Planned wall-clock budget reservations derived from `T0` and `PipelineConfig`."""

    t0: float
    session_sec: float
    safety_margin_sec: float
    setup_max_sec: float
    preflight_max_sec: float
    export_sec: float
    package_sec: float
    eval_sec: float
    train_deadline: float
    eval_deadline: float
    session_hard_limit: float


def resolve_session_t0(cfg: PipelineConfig, explicit_t0: float | None = None, now_fn: Callable[[], float] = time.time) -> float:
    """Resolves `T0` from explicit arg, `cfg.run.session_start_env`, or `now_fn()`."""
    if explicit_t0 is not None:
        return float(explicit_t0)
    env_name = cfg.run.session_start_env
    if env_name and os.environ.get(env_name):
        try:
            return float(os.environ[env_name])
        except ValueError:
            pass
    return float(now_fn())


class SessionClock:
    """Tracks elapsed session time, computes phase deadlines, and splits adapter training windows."""

    def __init__(
        self,
        cfg: PipelineConfig,
        t0: float | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self._now_fn = now_fn
        self.t0 = resolve_session_t0(cfg, explicit_t0=t0, now_fn=now_fn)
        self.plan = self._build_plan()

    def now(self) -> float:
        return float(self._now_fn())

    def elapsed(self) -> float:
        return max(0.0, self.now() - self.t0)

    def remaining_until(self, deadline: float) -> float:
        return max(0.0, deadline - self.now())

    def remaining_train_sec(self) -> float:
        return self.remaining_until(self.plan.train_deadline)

    def _build_plan(self) -> PhasePlan:
        b = self.cfg.budget
        eval_enabled = self.cfg.eval.enabled

        if self.cfg.run.mode == "calibration":
            session_sec = self.cfg.run.calibration_minutes * 60.0
            raw_margin = b.safety_margin_min * 60.0
            raw_eval = (b.eval_min * 60.0) if eval_enabled else 0.0
            raw_pkg = b.package_min * 60.0
            raw_exp = b.export_min * 60.0
            raw_setup = b.setup_max_min * 60.0
            raw_preflight = b.preflight_max_min * 60.0

            raw_reserved = raw_margin + raw_eval + raw_pkg + raw_exp
            # Ensure at least 45% of calibration_minutes is available for setup + preflight + train.
            max_post_train = session_sec * (0.55 if eval_enabled else 0.25)
            if raw_reserved > max_post_train and raw_reserved > 0:
                scale = max_post_train / raw_reserved
                margin_sec = raw_margin * scale
                eval_sec = raw_eval * scale
                pkg_sec = raw_pkg * scale
                exp_sec = raw_exp * scale
            else:
                margin_sec, eval_sec, pkg_sec, exp_sec = raw_margin, raw_eval, raw_pkg, raw_exp

            train_window = session_sec - (margin_sec + eval_sec + pkg_sec + exp_sec)
            setup_max_sec = min(raw_setup, max(30.0, train_window * 0.25))
            preflight_max_sec = min(raw_preflight, max(30.0, train_window * 0.30))
        else:
            session_sec = b.session_hours * 3600.0
            margin_sec = b.safety_margin_min * 60.0
            eval_sec = (b.eval_min * 60.0) if eval_enabled else 0.0
            pkg_sec = b.package_min * 60.0
            exp_sec = b.export_min * 60.0
            setup_max_sec = b.setup_max_min * 60.0
            preflight_max_sec = b.preflight_max_min * 60.0

        session_hard_limit = self.t0 + session_sec
        eval_deadline = session_hard_limit - margin_sec
        train_deadline = eval_deadline - eval_sec - pkg_sec - exp_sec

        return PhasePlan(
            t0=self.t0,
            session_sec=session_sec,
            safety_margin_sec=margin_sec,
            setup_max_sec=setup_max_sec,
            preflight_max_sec=preflight_max_sec,
            export_sec=exp_sec,
            package_sec=pkg_sec,
            eval_sec=eval_sec,
            train_deadline=train_deadline,
            eval_deadline=eval_deadline,
            session_hard_limit=session_hard_limit,
        )

    def phase_deadline(self, phase: str) -> float:
        """Computes the hard deadline for a non-training phase launched at `self.now()`."""
        now = self.now()
        p = self.plan
        if phase == "setup":
            return min(now + p.setup_max_sec, p.train_deadline)
        if phase == "preflight":
            return min(now + p.preflight_max_sec, p.train_deadline)
        if phase == "export":
            target_cap = p.eval_deadline - p.eval_sec - p.package_sec
            return max(now + min(60.0, p.export_sec), min(now + p.export_sec, target_cap))
        if phase == "package":
            # Always give `package` its budget so G4 ("Always ship") never starves.
            return min(now + max(60.0, p.package_sec), p.session_hard_limit)
        if phase == "eval":
            return p.eval_deadline
        if phase == "report":
            return p.session_hard_limit
        raise ValueError(f"Unknown phase {phase!r}")

    def compute_adapter_weights(
        self,
        adapters: list[AdapterSpec],
        supervised_tokens_by_adapter: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Computes normalized time-share weights across `adapters` per design_doc.md §7.7."""
        if not adapters:
            return {}

        sup_map = supervised_tokens_by_adapter or {}
        explicit = {a.name: a.time_share for a in adapters if a.time_share is not None}
        unset = [a for a in adapters if a.time_share is None]

        explicit_sum = sum(explicit.values())
        rem_share = max(0.0, 1.0 - explicit_sum)

        raw_unset: dict[str, float] = {}
        for a in unset:
            eff = self.cfg.resolve_adapter(a)
            toks = max(1, int(sup_map.get(a.name, 1)))
            raw_unset[a.name] = float(toks) * float(eff.train.max_epochs)

        unset_total = sum(raw_unset.values())
        weights: dict[str, float] = {}
        for a in adapters:
            if a.name in explicit:
                weights[a.name] = float(explicit[a.name])
            elif unset_total > 0 and rem_share > 0:
                weights[a.name] = rem_share * (raw_unset[a.name] / unset_total)
            else:
                weights[a.name] = 1.0 / len(adapters)

        total_w = sum(weights.values())
        if total_w <= 0:
            return {a.name: 1.0 / len(adapters) for a in adapters}
        return {k: v / total_w for k, v in weights.items()}

    def next_adapter_deadline(
        self,
        current_adapter: AdapterSpec,
        remaining_adapters: list[AdapterSpec],
        supervised_tokens_by_adapter: dict[str, int] | None = None,
    ) -> float:
        """Returns the wall-clock deadline for `current_adapter`, redistributing any prior leftover time.

        Because `remaining_train_sec()` is evaluated at `self.now()`, any time saved by an earlier
        adapter finishing early (or being skipped) is automatically pooled into the remaining window
        and split proportionally among `remaining_adapters`.
        """
        now = self.now()
        window = max(0.0, self.plan.train_deadline - now)
        if window <= 0.0 or not remaining_adapters:
            return now

        shares = self.compute_adapter_weights(remaining_adapters, supervised_tokens_by_adapter)
        share = shares.get(current_adapter.name, 1.0 / len(remaining_adapters))
        return now + window * share
