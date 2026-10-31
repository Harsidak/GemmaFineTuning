"""End-to-end test of the Tunix backend on a tiny Gemma 4 checkpoint, on JAX-CPU.

Needs the JAX stack (jax, flax, qwix, tunix, optax), which the repo's `.venv` does not install, so it is skipped there.
Run it in an environment that has the TPU wheel set (or any JAX-CPU install of the same packages):

    TUNIX_TINY_FIXTURE=<dir from save_tiny_gemma4_checkpoint(vocab_size=512, num_hidden_layers=4, num_kv_shared_layers=2)>
    JAX_PLATFORMS=cpu python -m pytest tpu_training/tests/test_tunix_integration.py
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("qwix")
pytest.importorskip("tunix")
pytest.importorskip("optax")

FIXTURE = os.environ.get("TUNIX_TINY_FIXTURE")
pytestmark = pytest.mark.skipif(not FIXTURE or not Path(FIXTURE).is_dir(), reason="set TUNIX_TINY_FIXTURE to a tiny Gemma 4 dir")

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.runctx import RunContext

REPO = Path(__file__).resolve().parents[2]
ADAPTER = "main_lora"


def _rows(n: int, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        ln = rnd.randint(18, 60)
        ids = [rnd.randint(3, 500) for _ in range(ln)]
        labels = [-100] * (ln // 2) + ids[ln // 2:]
        rows.append({"task_id": f"t{i:03d}", "input_ids": ids, "labels": labels, "weights": [1.0] * ln, "length": ln})
    return rows


def _make_run(tmp: Path, run_id: str, n_train: int = 24, extra: list[str] | None = None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = load_config(
        REPO / "pipeline" / "multi_lora_sft" / "configs" / "e4b_tpu.yaml",
        set_overrides=[
            f"run.output_root={(tmp / 'runs').as_posix()}", f"run.run_id={run_id}",
            f"model.train_base_path={Path(FIXTURE).as_posix()}", f"model.served_path={Path(FIXTURE).as_posix()}",
            "model.quantization=fp32", "train.backend=tunix", "train.accelerator=cpu", "train.parallelism=single",
            "train.per_device_batch=2", "train.grad_accum=2", "train.max_epochs=1", "train.max_seq_length=128",
            "train.pad_to_multiple_of=16", "train.log_every_steps=1", "train.dead_check_step=3", "train.throughput_project_step=2",
            "train.checkpoint_every_min=0.0001", "train.val_every_min=0.0001", "train.longest_first_steps=1",
            "loss.chunk_rows=32", "lora.r=4", "lora.alpha=8", "schedule.warmup_steps=2", "tracking.wandb=off",
            *(extra or []),
        ],
    )
    ctx = RunContext(tmp / "runs" / run_id)
    ctx.ensure_dirs()
    d = ctx.cache_dir / ADAPTER
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(_rows(n_train)), d / "train.parquet")
    pq.write_table(pa.Table.from_pylist(_rows(4, seed=99)), d / "val.parquet")
    return cfg, ctx


def _run(cfg, ctx, deadline_sec: float = 600.0):
    import time

    from tpu_training.phase import run_tunix_adapter_phase

    return run_tunix_adapter_phase(ctx, ADAPTER, deadline=time.time() + deadline_sec, cfg=cfg)


def _metrics(ctx) -> list[dict]:
    p = ctx.metrics_dir / f"{ADAPTER}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def test_fresh_run_trains_checkpoints_and_exports_a_valid_adapter(tmp_path: Path) -> None:
    # 16 examples / (2 x 2) per step = 4 steps per epoch; 4 epochs over the same few examples so the loss visibly falls.
    cfg, ctx = _make_run(tmp_path, "fresh", n_train=16, extra=["train.max_epochs=4", "train.lr=3.0e-3"])
    out = _run(cfg, ctx)
    assert out["state"] == "ok"
    s = out["outputs"]
    assert s["backend"] == "tunix" and s["steps_completed"] == 16
    final = ctx.checkpoints_dir / ADAPTER / "final_adapter"
    assert (final / "adapter_model.safetensors").is_file() and (final / "adapter_config.json").is_file()
    assert s["session"]["contract"]["ok"] is True, s["session"]["contract"]
    assert s["dead_lora_check"]["dead_lora_b_modules"] == 0
    losses = [m["train_loss"] for m in _metrics(ctx)]
    assert len(losses) >= 4 and all(math.isfinite(x) for x in losses) and losses[-1] < losses[0]
    assert any(p.name.startswith("step_") for p in (ctx.checkpoints_dir / ADAPTER).iterdir())
    assert (ctx.checkpoints_dir / ADAPTER / "best_adapter").is_dir()  # validation ran on the wall-clock timer


def test_resume_continues_from_the_ledger_and_restores_optimizer_state(tmp_path: Path) -> None:
    cfg, ctx = _make_run(tmp_path, "resume", extra=["train.max_steps=3", "train.max_epochs=3"])
    first = _run(cfg, ctx)["outputs"]
    assert first["steps_completed"] == 3
    # Same run directory, bigger target: a crash-restart style resume from this run's newest checkpoint.
    cfg2, _ = _make_run(tmp_path, "resume", extra=["train.max_steps=6", "train.max_epochs=3", "resume.extend=true"])
    second = _run(cfg2, ctx)["outputs"]
    assert second["resumed_from"] is not None
    assert second["resume"]["optimizer_restored"] is True
    assert second["resume"]["data_plan"], "task ledger plan expected"
    assert second["steps_completed"] == 6 and second["segment_steps"] == 3
    led = json.loads((ctx.checkpoints_dir / ADAPTER / "progress.json").read_text())
    assert led["sessions"], "ledger keeps a per-session history"


def test_a_checkpoint_the_model_cannot_load_falls_back_to_a_fresh_lora_without_skipping_data(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import load_file, save_file

    cfg, ctx = _make_run(tmp_path, "badckpt", extra=["train.max_steps=2"])
    _run(cfg, ctx)
    ck = max((ctx.checkpoints_dir / ADAPTER).glob("step_*")) / "adapter" / "adapter_model.safetensors"
    t = load_file(str(ck))
    t.pop(next(k for k in t if k.endswith("mlp.up_proj.lora_B.weight")))
    save_file({k: np.asarray(v) for k, v in t.items()}, str(ck))  # structurally damaged but still a valid safetensors file
    cfg2, _ = _make_run(tmp_path, "badckpt", extra=["train.max_steps=2", "resume.extend=true"])
    out = _run(cfg2, ctx)["outputs"]
    assert out["resumed_from"] is None
    assert any("cannot be loaded" in n for n in out["resume"]["notes"])


def test_non_finite_gradients_skip_the_update_and_eventually_abort(tmp_path: Path, monkeypatch) -> None:
    from pipeline.multi_lora_sft.shared.guards import NonFiniteLossError
    from tpu_training import trainer as tr

    cfg, ctx = _make_run(tmp_path, "nan", extra=["train.max_nonfinite_logs=2"])
    orig = tr.LoraTrainer.step

    def bad_step(self, mbs, lr):  # a step whose loss and gradient are NaN: weights must not move, guard must trip
        res = orig(self, mbs, lr)
        return {**res, "loss": float("nan"), "grad_norm": float("nan"), "applied": False}

    monkeypatch.setattr(tr.LoraTrainer, "step", bad_step)
    with pytest.raises(NonFiniteLossError):
        _run(cfg, ctx)
    status = json.loads((ctx.status_path(f"train:{ADAPTER}")).read_text())
    assert status["state"] == "failed"
    assert not list((ctx.checkpoints_dir / ADAPTER).glob("step_*")), "never checkpoint after non-finite steps"
