"""Edge cases: model-dir validation, TPU config rules, static-shape loss, NaN guard, resume step offsets,
disk-aware / corrupt checkpoints, per-attempt OOM detection, and the submission's generation/eval configs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn
from transformers import TrainerControl, TrainerState

from pipeline.multi_lora_sft.config import ConfigError, load_config
from pipeline.multi_lora_sft.phases.package import write_generation_and_eval_configs
from pipeline.multi_lora_sft.run import _detect_oom
from pipeline.multi_lora_sft.training import callbacks as cb_mod
from pipeline.multi_lora_sft.training.callbacks import (
    MultiLoraTrainingCallback,
    NonFiniteLossError,
    checkpoint_is_loadable,
    save_wallclock_checkpoint,
)
from pipeline.multi_lora_sft.training.data import LengthGroupedLongestFirstSampler
from pipeline.multi_lora_sft.training.device import (
    AcceleratorUnavailableError,
    is_oom_text,
    resolve_accelerator,
    xla_installed,
)
from pipeline.multi_lora_sft.training.loss import (
    WeightedLanguageModelingCollator,
    selective_cross_entropy_loss,
    static_cross_entropy_loss,
)
from pipeline.multi_lora_sft.training.model import (
    ModelLoadError,
    check_train_served_compat,
    validate_model_dir,
)
from pipeline.multi_lora_sft.training.schedule import WSDScheduleController

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ----------------------------------------------------------------------------- config rules


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        (["train.accelerator=tpu", "train.parallelism=spmd", "model.quantization=nf4"], "nf4"),
        (["train.accelerator=tpu", "train.parallelism=ddp", "model.quantization=bf16"], "CUDA-only"),
        (["train.accelerator=cuda", "train.parallelism=spmd"], "requires accelerator 'tpu'"),
        (["submission.generation.thinking_budget=16384"], "thinking_budget must be <"),
        (["submission.eval_budget.max_turns=501"], "max_turns"),
        (["submission.generation.top_p=0"], "top_p"),
        (["train.accelerator=gpu"], "accelerator must be"),
    ],
)
def test_invalid_hardware_and_submission_settings_rejected(overrides: list[str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(None, overrides)


def test_tpu_preset_config_loads() -> None:
    cfg = load_config(CONFIGS / "e4b_tpu.yaml")
    eff = cfg.resolve_adapter("main_lora")
    assert (eff.train.accelerator, eff.train.parallelism, eff.loss.impl) == ("tpu", "spmd", "static")
    assert eff.model_quantization == "bf16"
    assert cfg.eval.enabled is False


# ----------------------------------------------------------------------------- model directory checks


def _write_ckpt(root: Path, config: dict, shards: dict[str, bytes] | None = None, index: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name, data in (shards or {}).items():
        (root / name).write_bytes(data)
    if index is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_validate_model_dir_reports_actionable_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="does not exist"):
        validate_model_dir(tmp_path / "nope", "model.train_base_path", for_training=True)

    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelLoadError, match="no config.json"):
        validate_model_dir(tmp_path / "empty", "model.train_base_path", for_training=True)

    ct = _write_ckpt(
        tmp_path / "w4a16",
        {"model_type": "gemma4", "quantization_config": {"quant_method": "compressed-tensors"}},
        {"model.safetensors": b"x"},
    )
    with pytest.raises(ModelLoadError, match="unquantized"):
        validate_model_dir(ct, "model.train_base_path", for_training=True)
    assert validate_model_dir(ct, "model.served_path", for_training=False)["model_type"] == "gemma4"

    partial = _write_ckpt(
        tmp_path / "partial",
        {"model_type": "gemma4"},
        {"model-00001-of-00002.safetensors": b"x"},
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    with pytest.raises(ModelLoadError, match="incomplete"):
        validate_model_dir(partial, "model.train_base_path", for_training=True)

    no_weights = _write_ckpt(tmp_path / "noweights", {"model_type": "gemma4"})
    with pytest.raises(ModelLoadError, match="no non-empty weight files"):
        validate_model_dir(no_weights, "model.train_base_path", for_training=True)


def test_validate_model_dir_accepts_tiny_fixture(tiny_gemma4_dir: Path) -> None:
    cfg = validate_model_dir(tiny_gemma4_dir, "model.train_base_path", for_training=True)
    assert cfg["model_type"].startswith("gemma4")


def test_train_served_size_mismatch_is_fatal() -> None:
    big = {"text_config": {"hidden_size": 5376, "num_hidden_layers": 60}}
    small = {"text_config": {"hidden_size": 2560, "num_hidden_layers": 42}}
    check_train_served_compat(big, big)
    with pytest.raises(ModelLoadError, match="different model sizes"):
        check_train_served_compat(big, small)


# ----------------------------------------------------------------------------- loss / collator


def _loss_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    bsz, seq, hid, vocab = 2, 13, 8, 32
    hidden = torch.randn(bsz, seq, hid, generator=g, requires_grad=True)
    head = nn.Linear(hid, vocab, bias=False)
    labels = torch.randint(0, vocab, (bsz, seq), generator=g)
    labels[0, :4] = -100
    labels[1, 9:] = -100
    weights = torch.rand(bsz, seq, generator=g) + 0.5
    return hidden, head, labels, weights


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_static_loss_matches_selective_loss_and_grads(checkpoint_chunks: bool) -> None:
    hidden, head, labels, weights = _loss_inputs()
    kw = {"weights": weights, "softcap": 30.0, "chunk_rows": 6, "checkpoint_chunks": checkpoint_chunks}

    sel = selective_cross_entropy_loss(hidden, head, labels, **kw)
    (g_sel,) = torch.autograd.grad(sel, hidden)
    stat = static_cross_entropy_loss(hidden, head, labels, **kw)
    (g_stat,) = torch.autograd.grad(stat, hidden)

    assert torch.allclose(sel, stat, atol=1e-5)
    assert torch.allclose(g_sel, g_stat, atol=1e-5)


def test_static_loss_all_masked_is_zero_not_nan() -> None:
    hidden, head, labels, _ = _loss_inputs()
    labels.fill_(-100)
    loss = static_cross_entropy_loss(hidden, head, labels)
    assert loss.item() == 0.0


def test_collator_pads_to_multiple_and_rejects_empty_batch() -> None:
    coll = WeightedLanguageModelingCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = coll([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4] * 9, "labels": [4] * 9}])
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"][0, 3:].eq(-100).all()
    with pytest.raises(ValueError, match="empty batch"):
        coll([])


# ----------------------------------------------------------------------------- callback: offsets, NaN, stop


class _TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Linear(4, 2, bias=False)

    def save_pretrained(self, out_dir: Path, **_: object) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file({"lora_A.weight": self.lora_A.weight.detach().contiguous()}, str(out / "adapter_model.safetensors"))


class _FakeTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _s: 1.0)


def _callback(tmp_path: Path, *, offset: int = 0, stop=None, sampler=None, ckpt_min: float = 1e-9):
    now = time.time()
    wsd = WSDScheduleController(total_epoch_steps=1000, t_start=now, deadline=now + 3600, warmup_steps=0)
    model = _TinyAdapterModel()
    cb = MultiLoraTrainingCallback(
        adapter_name="main_lora",
        wsd_controller=wsd,
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
        checkpoint_root=tmp_path / "ckpt",
        metrics_tracker={},
        sampler=sampler,
        dataset_total_tokens=100,
        samples_per_step=4,
        checkpoint_every_min=ckpt_min,
        step_offset=offset,
        max_nonfinite_logs=3,
        stop_requested=stop,
    )
    cb.bind_trainer(_FakeTrainer(model))
    if ckpt_min < 1.0:
        cb.ckpt_every_sec = 0.0  # the callback floors the interval at 1 s; tests checkpoint every step
    return cb, model, wsd


ARGS = SimpleNamespace(world_size=1)


def test_resumed_checkpoints_use_global_step_numbers(tmp_path: Path) -> None:
    sampler = LengthGroupedLongestFirstSampler([5] * 40, seed=1)
    sampler.load_state_dict({"epoch": 0, "offset": 12})
    cb, model, wsd = _callback(tmp_path, offset=100, sampler=sampler)
    state, control = TrainerState(), TrainerControl()
    cb.on_epoch_begin(ARGS, state, control)
    state.global_step = 5
    cb.on_step_end(ARGS, state, control, model=model)

    saved = sorted((tmp_path / "ckpt").glob("step_*"))
    assert [p.name for p in saved] == ["step_000105"]
    meta = json.loads((saved[0] / "trainer_resume.json").read_text(encoding="utf-8"))
    assert meta["step"] == 105
    assert meta["sampler_state"]["offset"] == 12 + 4  # resumed offset + one step of samples
    assert wsd.step == 105


def test_nonfinite_guard_raises_after_streak_and_resets_on_finite(tmp_path: Path) -> None:
    cb, _, _ = _callback(tmp_path, ckpt_min=1e9)
    state, control = TrainerState(), TrainerControl()
    cb.on_log(None, state, control, logs={"loss": float("nan")})
    cb.on_log(None, state, control, logs={"loss": 1.0, "grad_norm": 0.5})
    assert cb.nonfinite_streak == 0
    cb.on_log(None, state, control, logs={"loss": 1.0, "grad_norm": float("inf")})
    cb.on_log(None, state, control, logs={"loss": float("nan")})
    with pytest.raises(NonFiniteLossError):
        cb.on_log(None, state, control, logs={"loss": float("nan")})


def test_stop_request_ends_training_at_step_boundary(tmp_path: Path) -> None:
    cb, model, _ = _callback(tmp_path, stop=lambda: True, ckpt_min=1e9)
    state, control = TrainerState(), TrainerControl()
    state.global_step = 1
    cb.on_step_end(ARGS, state, control, model=model)
    assert control.should_training_stop and cb.stopped_by_signal


def test_checkpoint_skipped_not_crashing_when_disk_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cb, model, wsd = _callback(tmp_path, ckpt_min=1e9)

    class _Usage:
        free = 10 * 1024**2

    monkeypatch.setattr(cb_mod.shutil, "disk_usage", lambda _p: _Usage())
    out = save_wallclock_checkpoint(
        checkpoint_root=tmp_path / "ckpt",
        step=1,
        epoch=0.0,
        model=model,
        optimizer=None,
        scheduler=None,
        wsd_controller=wsd,
        sampler=None,
        samples_in_epoch=0,
        keep_checkpoints=2,
        min_free_disk_gib=1.0,
    )
    assert out is None
    assert not list((tmp_path / "ckpt").glob("step_*"))


def test_corrupt_newest_checkpoint_is_skipped_on_resume(tmp_path: Path) -> None:
    from pipeline.multi_lora_sft.phases.train import resolve_resume_checkpoint
    from pipeline.multi_lora_sft.runctx import RunContext

    cfg = load_config(None, [f"run.output_root={tmp_path.as_posix()}", "run.run_id=r1"])
    ctx = RunContext.create(cfg, time.time())
    root = ctx.checkpoints_dir / "main_lora"
    _cb, model, wsd = _callback(tmp_path)
    for step in (10, 20):
        save_wallclock_checkpoint(
            checkpoint_root=root, step=step, epoch=0.0, model=model, optimizer=None, scheduler=None,
            wsd_controller=wsd, sampler=None, samples_in_epoch=0, keep_checkpoints=5,
        )
    (root / "step_000020" / "adapter" / "adapter_model.safetensors").write_bytes(b"\x00truncated")
    assert not checkpoint_is_loadable(root / "step_000020")
    assert resolve_resume_checkpoint(ctx, cfg, "main_lora") == root / "step_000010"


# ----------------------------------------------------------------------------- orchestrator / device


def test_oom_detection_only_considers_latest_attempt(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "--- [t0] LAUNCH train:main_lora ---\nRuntimeError: CUDA out of memory\n"
        "--- [t1] LAUNCH train:main_lora ---\nValueError: something else\n",
        encoding="utf-8",
    )
    assert _detect_oom(log, {"error": "ValueError: something else"}) is False
    assert _detect_oom(log, {"error": "RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm"}) is True


def test_accelerator_resolution_and_oom_signatures() -> None:
    assert is_oom_text("Insufficient accelerator memory (out of memory): device_map offloaded 3 module(s)")
    assert not is_oom_text("KeyError: 'weights'")
    if not xla_installed():
        with pytest.raises(AcceleratorUnavailableError, match="torch_xla"):
            resolve_accelerator("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(AcceleratorUnavailableError):
            resolve_accelerator("cuda")
        assert resolve_accelerator("auto") in {"cpu", "tpu"}


# ----------------------------------------------------------------------------- packaged configs


def test_package_writes_generation_and_eval_budget(tmp_path: Path) -> None:
    sub = tmp_path / "submission"
    (sub / "configs").mkdir(parents=True)
    (sub / "agent.yaml").write_text("generate_content_config: !include configs/sampling.yaml\n", encoding="utf-8")
    (sub / "eval_config.yaml").write_text("evaluation: {max_time_minutes: 1, max_tool_calls: 10}\n", encoding="utf-8")
    cfg = load_config(
        None,
        ["submission.generation.temperature=0.6", "submission.generation.top_k=40", "submission.eval_budget.max_time_minutes=7"],
    )
    notes = write_generation_and_eval_configs(cfg, sub)
    assert notes == []
    sampling = yaml.safe_load((sub / "configs" / "sampling.yaml").read_text(encoding="utf-8"))
    assert sampling["temperature"] == 0.6 and sampling["top_k"] == 40
    assert sampling["thinking_config"] == {"thinking_budget": 4096, "include_thoughts": True}
    assert "thinking_level" not in json.dumps(sampling)
    evaluation = yaml.safe_load((sub / "eval_config.yaml").read_text(encoding="utf-8"))["evaluation"]
    assert evaluation["max_time_minutes"] == 7.0 and evaluation["max_tool_calls"] == 100
    assert math.isclose(evaluation["timeout_seconds"], 300)
