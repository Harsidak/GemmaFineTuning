"""Unit tests for `pipeline.multi_lora_sft.clock` (design_doc.md §5, §7.7, §11)."""

from __future__ import annotations

import pytest

from pipeline.multi_lora_sft.clock import SessionClock
from pipeline.multi_lora_sft.config import load_config


def test_12h_budget_deadlines_add_up() -> None:
    cfg = load_config()
    now_val = 1_000_000.0
    clock = SessionClock(cfg, t0=now_val, now_fn=lambda: now_val)

    # 12h = 43200s; reserved = 20m margin + 60m eval + 5m package + 10m export = 95m = 5700s -> 10h25m = 37500s
    assert clock.plan.session_sec == pytest.approx(12.0 * 3600.0)
    assert clock.plan.train_deadline - clock.t0 == pytest.approx((10 * 60 + 25) * 60.0)
    assert clock.plan.eval_deadline - clock.t0 == pytest.approx((11 * 60 + 40) * 60.0)
    assert clock.plan.session_hard_limit - clock.t0 == pytest.approx(12.0 * 3600.0)

    assert clock.phase_deadline("setup") - now_val == pytest.approx(10.0 * 60.0)
    assert clock.phase_deadline("preflight") - now_val == pytest.approx(15.0 * 60.0)


def test_leftover_time_redistribution_between_adapters() -> None:
    cfg = load_config(
        set_overrides=[
            "adapters=[{name: main_lora, roles: [root]}, {name: tool_lora, roles: [code_analyzer]}]"
        ]
    )
    cur = [0.0]
    clock = SessionClock(cfg, t0=0.0, now_fn=lambda: cur[0])
    total_train_sec = clock.plan.train_deadline  # 37500s

    # Preflight took 500s; remaining train window is 37000s
    cur[0] = 500.0
    sup_tokens = {"main_lora": 3000, "tool_lora": 1000}  # 75% vs 25%

    d1 = clock.next_adapter_deadline(cfg.adapters[0], cfg.adapters, sup_tokens)
    assert d1 - cur[0] == pytest.approx(37000.0 * 0.75)

    # Suppose main_lora finishes early in just 7000s instead of 27750s!
    cur[0] = 7500.0
    # Remaining adapter (tool_lora) should receive 100% of the remaining 30000s up to train_deadline
    d2 = clock.next_adapter_deadline(cfg.adapters[1], cfg.adapters[1:], sup_tokens)
    assert d2 == pytest.approx(total_train_sec)
    assert d2 - cur[0] == pytest.approx(30000.0)


def test_calibration_mode_scales_phase_budgets() -> None:
    cfg = load_config(
        set_overrides=[
            "run.mode=calibration",
            "run.calibration_minutes=30",
            "eval.enabled=false",
        ]
    )
    clock = SessionClock(cfg, t0=100.0, now_fn=lambda: 100.0)
    assert clock.plan.session_sec == pytest.approx(1800.0)
    # With eval disabled, at least 75% of 1800s (1350s) is available for setup+preflight+train
    assert clock.remaining_train_sec() >= 1350.0 - 1e-6
