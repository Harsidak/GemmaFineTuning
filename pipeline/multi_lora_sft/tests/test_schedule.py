"""Unit tests for `pipeline.multi_lora_sft.training.schedule` (design_doc.md §7.6, §7.8, §11)."""

from __future__ import annotations

import pytest

from pipeline.multi_lora_sft.training.schedule import WSDScheduleController


def test_warmup_stable_and_deadline_first_decay() -> None:
    cur = [0.0]
    ctrl = WSDScheduleController(
        warmup_steps=4,
        decay_frac=0.20,
        final_lr_ratio=0.10,
        ema_beta=0.5,
        total_epoch_steps=1000,  # epochs won't finish before deadline=100s when each step takes 1s
        t_start=0.0,
        deadline=100.0,
        now_fn=lambda: cur[0],
    )

    # Warmup steps 0, 1, 2, 3 -> 0.25, 0.50, 0.75, 1.00
    assert ctrl.lr_lambda(0) == pytest.approx(0.25)
    cur[0] = 1.0
    ctrl.on_step_end(1)
    assert ctrl.lr_lambda(1) == pytest.approx(0.50)

    cur[0] = 4.0
    ctrl.on_step_end(4)
    assert ctrl.lr_lambda(4) == pytest.approx(1.0)
    assert not ctrl.decay_started

    # Stable up to t=79s (decay starts at 100 - 0.20*100 = 80s)
    for s in range(5, 80):
        cur[0] = float(s)
        ctrl.on_step_end(s)
    assert not ctrl.decay_started
    assert ctrl.lr_lambda(79) == pytest.approx(1.0)

    # At t=80s, decay begins and freezes t_end=100.0
    cur[0] = 80.0
    ctrl.on_step_end(80)
    assert ctrl.decay_started
    assert ctrl.frozen_t_end == pytest.approx(100.0)

    # Halfway through decay (t=90s), even if a step suddenly took 5s, frozen_t_end stays 100.0
    cur[0] = 90.0
    ctrl.on_step_end(82)
    assert ctrl.frozen_t_end == pytest.approx(100.0)
    assert ctrl.lr_lambda(82) == pytest.approx(0.55)  # halfway between 1.0 and 0.1

    cur[0] = 100.0
    ctrl.on_step_end(85)
    assert ctrl.lr_lambda(85) == pytest.approx(0.10)
    assert ctrl.should_stop()


def test_epochs_first_decay_starts_early() -> None:
    cur = [0.0]
    # 20 total steps at 2s/step -> projected end is 40s, well before deadline=200s!
    ctrl = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.25,
        final_lr_ratio=0.10,
        ema_beta=0.5,
        total_epoch_steps=20,
        t_start=0.0,
        deadline=200.0,
        now_fn=lambda: cur[0],
    )

    for s in range(1, 15):
        cur[0] = float(s * 2)
        ctrl.on_step_end(s)
    assert not ctrl.decay_started  # t=28s < 40 * (1 - 0.25) = 30s

    cur[0] = 30.0
    ctrl.on_step_end(15)
    assert ctrl.decay_started
    assert ctrl.frozen_t_end == pytest.approx(40.0)


def test_resume_mid_stable_and_mid_decay() -> None:
    cur = [0.0]
    ctrl1 = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.20,
        final_lr_ratio=0.10,
        total_epoch_steps=500,
        t_start=0.0,
        deadline=100.0,
        now_fn=lambda: cur[0],
    )
    for s in range(1, 41):
        cur[0] = float(s)
        ctrl1.on_step_end(s)
        ctrl1.lr_lambda(s)

    # 1. Save mid-stable at t=40s
    mid_stable_state = ctrl1.state_dict()
    assert not mid_stable_state["decay_started"]

    cur2 = [1000.0]
    resumed_stable = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.20,
        final_lr_ratio=0.10,
        total_epoch_steps=500,
        t_start=1000.0,
        deadline=1100.0,
        now_fn=lambda: cur2[0],
    )
    resumed_stable.load_resume_state(mid_stable_state, new_t_start=1000.0, new_deadline=1100.0, extend=False)
    assert not resumed_stable.decay_started
    assert resumed_stable.lr_lambda(40) == pytest.approx(1.0)

    # 2. Advance ctrl1 into mid-decay at t=90s (multiplier = 0.55)
    for s in range(41, 91):
        cur[0] = float(s)
        ctrl1.on_step_end(s)
        ctrl1.lr_lambda(s)
    mid_decay_state = ctrl1.state_dict()
    assert mid_decay_state["decay_started"]

    # Resume mid-decay with extend=False over a 200s new budget -> finishes decay within <= 10% (<= 20s, capped by remaining 10s) and stops!
    cur3 = [2000.0]
    resumed_decay = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.20,
        final_lr_ratio=0.10,
        total_epoch_steps=500,
        t_start=2000.0,
        deadline=2200.0,
        now_fn=lambda: cur3[0],
    )
    resumed_decay.load_resume_state(mid_decay_state, new_t_start=2000.0, new_deadline=2200.0, extend=False)
    assert resumed_decay.lr_lambda(90) == pytest.approx(0.55)
    cur3[0] = 2010.0
    resumed_decay.on_step_end(95)
    assert resumed_decay.lr_lambda(95) == pytest.approx(0.10)
    assert resumed_decay.should_stop()

    # Resume mid-decay with extend=True -> re-enters stable after quick-decay!
    cur4 = [3000.0]
    resumed_ext = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.20,
        final_lr_ratio=0.10,
        total_epoch_steps=500,
        t_start=3000.0,
        deadline=3200.0,
        now_fn=lambda: cur4[0],
    )
    resumed_ext.load_resume_state(mid_decay_state, new_t_start=3000.0, new_deadline=3200.0, extend=True)
    cur4[0] = 3011.0
    resumed_ext.on_step_end(95)
    assert not resumed_ext.should_stop()
    assert resumed_ext.lr_lambda(95) == pytest.approx(1.0)
