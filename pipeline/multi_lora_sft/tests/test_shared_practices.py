"""Backend-agnostic practices (pipeline/multi_lora_sft/shared): checkpoints, guards, resume compatibility, the task
ledger, tracking policy, secret scanning, the adapter contract, the cross-session bundle and the Optuna helpers.

Nothing here needs torch or jax, so these tests also guard the Tunix path."""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from pipeline.multi_lora_sft.config import ConfigError, load_config
from pipeline.multi_lora_sft.shared import fsio
from pipeline.multi_lora_sft.shared.adapter_contract import check_adapter_contract
from pipeline.multi_lora_sft.shared.backend import resolve_backend
from pipeline.multi_lora_sft.shared.bundle import (
    best_params_from_bundles,
    discover_bundles,
    resume_candidates,
    sync_bundle,
    tuning_source,
)
from pipeline.multi_lora_sft.shared.checkpoints import (
    CheckpointError,
    checkpoint_is_loadable,
    list_checkpoints,
    read_best_adapter,
    read_safetensors_header,
    write_best_adapter,
    write_checkpoint,
)
from pipeline.multi_lora_sft.shared.compat import (
    ResumeIncompatible,
    compare_fingerprints,
    data_fingerprint,
    enforce,
    training_fingerprint,
)
from pipeline.multi_lora_sft.shared.guards import (
    BestTracker,
    LossDivergedError,
    LossSpikeGuard,
    NonFiniteGuard,
    NonFiniteLossError,
)
from pipeline.multi_lora_sft.shared.ledger import ProgressLedger, example_key, example_keys
from pipeline.multi_lora_sft.shared.secrets_scan import scan_dir, scan_text
from pipeline.multi_lora_sft.shared.tracking import Tracker, decide_tracking
from pipeline.multi_lora_sft.shared.tuning import (
    DEFAULT_SPACE,
    TuningError,
    apply_best_params,
    build_trial_sets,
    params_to_sets,
    strip_tuned_overrides,
    trial_objective,
    validate_space,
)
from pipeline.multi_lora_sft.training.data import LengthGroupedLongestFirstSampler


# ----------------------------------------------------------------------------------------------- helpers
def write_adapter(path: Path, *, r: int = 4, layers: int = 2, mods=("self_attn.q_proj", "mlp.up_proj"), dims=(16, 32), b_scale=0.1) -> Path:
    """A PEFT-format adapter file written with numpy only."""
    path.mkdir(parents=True, exist_ok=True)
    t: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for layer in range(layers):
        for mod in mods:
            out_f, in_f = (dims[0], dims[0]) if "q_proj" in mod else (dims[1], dims[0])
            base = f"base_model.model.model.language_model.layers.{layer}.{mod}"
            t[f"{base}.lora_A.weight"] = rng.standard_normal((r, in_f)).astype(np.float32)
            t[f"{base}.lora_B.weight"] = (rng.standard_normal((out_f, r)) * b_scale).astype(np.float32)
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": 2 * r}), encoding="utf-8")
    return path


def fake_checkpoint_writer(step: int, r: int = 4):
    def _write(tmp: Path) -> None:
        write_adapter(tmp / "adapter", r=r)
        (tmp / "trainer_resume.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    return _write


# ----------------------------------------------------------------------------------------------- fsio
def test_atomic_write_leaves_no_partial_file_and_read_json_tolerates_corruption(tmp_path: Path) -> None:
    f = tmp_path / "a" / "x.json"
    fsio.atomic_write_json(f, {"k": 1})
    assert json.loads(f.read_text()) == {"k": 1}
    assert not [p for p in f.parent.iterdir() if p.name.endswith(".tmp")]
    f.write_text("{not json", encoding="utf-8")
    assert fsio.read_json(f, {"d": 1}) == {"d": 1}
    assert fsio.read_json(tmp_path / "missing.json", 5) == 5


def test_retry_gives_up_after_attempts_and_publish_dir_restores_on_failure(tmp_path: Path) -> None:
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("disk hiccup")
        return "ok"

    assert fsio.retry(flaky, attempts=3, delay=0, sleep=lambda _s: None) == "ok"
    with pytest.raises(OSError):
        fsio.retry(lambda: (_ for _ in ()).throw(OSError("x")), attempts=2, delay=0, sleep=lambda _s: None)

    final, tmp = tmp_path / "final", tmp_path / "tmp"
    final.mkdir()
    (final / "old.txt").write_text("old")
    tmp.mkdir()
    (tmp / "new.txt").write_text("new")
    fsio.publish_dir(tmp, final)
    assert (final / "new.txt").exists() and not (final / "old.txt").exists()


# ----------------------------------------------------------------------------------------------- checkpoints
def test_safetensors_header_detects_truncation(tmp_path: Path) -> None:
    p = write_adapter(tmp_path / "ad") / "adapter_model.safetensors"
    assert read_safetensors_header(p)["tensors"]
    data = p.read_bytes()
    p.write_bytes(data[: len(data) - 40])
    with pytest.raises(ValueError):
        read_safetensors_header(p)
    p.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError):
        read_safetensors_header(p)


def test_write_checkpoint_publishes_atomically_prunes_and_protects(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    for s in (10, 20, 30):
        out = write_checkpoint(root, s, fake_checkpoint_writer(s), keep=2)
        assert out is not None and checkpoint_is_loadable(out)
    assert [p.name for p in list_checkpoints(root)] == ["step_000020", "step_000030"]
    assert not list(root.glob(".tmp_*"))
    kept = write_checkpoint(root, 40, fake_checkpoint_writer(40), keep=1, protect=[root / "step_000030"])
    assert (root / "step_000030").exists() and kept is not None


def test_write_checkpoint_failure_keeps_previous_good_one(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    write_checkpoint(root, 10, fake_checkpoint_writer(10), keep=2)

    def boom(tmp: Path) -> None:
        write_adapter(tmp / "adapter")
        raise RuntimeError("serialisation exploded")

    with pytest.raises(RuntimeError):
        write_checkpoint(root, 20, boom, keep=2)
    assert [p.name for p in list_checkpoints(root)] == ["step_000010"]
    assert not list(root.glob(".tmp_*"))

    def no_meta(tmp: Path) -> None:  # missing trainer_resume.json -> verification must reject it
        write_adapter(tmp / "adapter")

    with pytest.raises(CheckpointError):
        write_checkpoint(root, 30, no_meta, keep=2)
    assert [p.name for p in list_checkpoints(root)] == ["step_000010"]


def test_write_checkpoint_skips_when_disk_is_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsio.shutil, "disk_usage", lambda _p: SimpleNamespace(free=10 * 1024**2))
    assert write_checkpoint(tmp_path / "c", 1, fake_checkpoint_writer(1), min_free_disk_gib=1.0) is None
    assert not list_checkpoints(tmp_path / "c")


def test_best_adapter_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    write_best_adapter(root, 50, 0.42, lambda d: write_adapter(d))
    got = read_best_adapter(root)
    assert got is not None and got[1]["step"] == 50 and got[1]["val_loss"] == pytest.approx(0.42)
    with pytest.raises(CheckpointError):
        write_best_adapter(root, 60, 0.4, lambda d: None)
    assert read_best_adapter(root)[1]["step"] == 50  # the failed write did not clobber it


# ----------------------------------------------------------------------------------------------- guards
def test_nonfinite_guard_streak_and_reset() -> None:
    g = NonFiniteGuard(3, "[t]")
    assert g.observe(float("nan")) and g.observe(1.0, 0.1) is False and g.streak == 0
    g.observe(1.0, float("inf"))
    g.observe(float("nan"))
    with pytest.raises(NonFiniteLossError):
        g.observe(float("nan"))


def test_loss_spike_guard_ignores_warmup_and_aborts_on_divergence() -> None:
    g = LossSpikeGuard(factor=4.0, limit=3, warmup_logs=5)
    for _ in range(10):
        g.observe(1.0)
    assert g.observe(50.0) is True and g.streak == 1
    g.observe(1.0)
    assert g.streak == 0
    with pytest.raises(LossDivergedError):
        for _ in range(3):
            g.observe(100.0)
    assert LossSpikeGuard(factor=0.0).observe(1e9) is False  # disabled


def test_best_tracker_early_stopping_and_state_roundtrip() -> None:
    b = BestTracker(patience=2, min_delta=0.01)
    assert b.update(10, 1.0).is_best
    assert not b.update(20, 0.995).is_best  # inside min_delta
    d = b.update(30, 1.2)
    assert d.should_stop and d.best_step == 10
    b2 = BestTracker(patience=2)
    b2.load_state_dict(b.state_dict())
    assert b2.best_loss == 1.0 and b2.stale == 2
    assert not BestTracker(patience=0).update(1, float("nan")).should_stop


# ----------------------------------------------------------------------------------------------- compat
def _fp(**over):
    base = dict(adapter="main_lora", served_model_id="gemma-4-e4b-it-qat-w4a16-ct", train_base="m/2", quantization="bf16",
                backend="trl", lora={"r": 32, "alpha": 64, "target_regex": "x", "use_rslora": False, "exclude_kv_shared": True})
    base.update(over)
    return training_fingerprint(**base)


def test_fingerprint_fatal_vs_warning() -> None:
    cur = _fp()
    assert compare_fingerprints(cur, cur).problems == []
    v = compare_fingerprints(_fp(lora={**cur["lora"], "r": 16}), cur)
    assert [p.code for p in v.fatal] == ["lora_r"]
    with pytest.raises(ResumeIncompatible):
        enforce(v, strict=True)
    assert enforce(v, strict=False) is v
    v2 = compare_fingerprints(_fp(backend="tunix", quantization="fp32"), cur)
    assert not v2.fatal and not v2.optimizer_compatible and {p.code for p in v2.warnings} == {"backend", "quantization"}
    assert compare_fingerprints(None, cur).warnings[0].code == "legacy_checkpoint"
    assert compare_fingerprints(_fp(served_model_id="gemma-4-31b-it-qat-w4a16-ct"), cur).fatal


def test_data_fingerprint_is_order_independent() -> None:
    assert data_fingerprint(["a", "b"])["keys_sha"] == data_fingerprint(["b", "a"])["keys_sha"]
    assert data_fingerprint(["a"])["keys_sha"] != data_fingerprint(["a", "b"])["keys_sha"]


# ----------------------------------------------------------------------------------------------- ledger
def _rows(n: int) -> list[dict]:
    return [{"task_id": f"t{i}", "input_ids": [i, i + 1, i + 2], "length": 3} for i in range(n)]


def test_example_key_is_stable_and_content_addressed() -> None:
    r = _rows(3)
    assert example_key(r[0]) == example_key(dict(r[0]))
    assert example_key(r[0]) != example_key({**r[0], "input_ids": [9, 9, 9]})
    assert example_key(r[1]).startswith("t1:")


def test_ledger_plan_continue_after_data_grew() -> None:
    old = example_keys(_rows(10))
    led = ProgressLedger("main_lora")
    led.snapshot(epoch=0, consumed_keys=old[:4], data_sha="x", data_count=10)
    grown = example_keys(_rows(14))  # 4 new problems were mined
    plan = led.plan(grown)
    assert plan.mode == "continue" and plan.done_in_epoch == 4 and plan.unseen == 10
    assert plan.exclude_indices == {0, 1, 2, 3}
    led.snapshot(epoch=0, consumed_keys=old, data_sha="x", data_count=10)
    assert led.plan(old).mode == "next_epoch" and led.plan(old).epoch == 1
    assert led.plan(grown, on_data_change="restart_epoch").mode == "restart"
    assert ProgressLedger("a").plan(grown).mode == "fresh"


def test_ledger_roundtrip_and_stale_keys(tmp_path: Path) -> None:
    led = ProgressLedger("main_lora")
    led.snapshot(epoch=2, consumed_keys=["gone:1", "t0:abc"], data_sha="s", data_count=5)
    led.save(tmp_path)
    again = ProgressLedger.load("main_lora", tmp_path)
    assert again is not None and again.epoch == 2 and again.consumed == {"gone:1", "t0:abc"}
    assert ProgressLedger.load("main_lora", tmp_path / "nope") is None
    plan = again.plan(["t0:abc", "t1:def"])
    assert plan.stale_keys == 1 and plan.exclude_indices == {0}


def test_sampler_exclusion_resumes_exactly_where_the_ledger_says() -> None:
    n = 40
    s = LengthGroupedLongestFirstSampler([5] * n, seed=7)
    full = list(iter(s))
    consumed = 15
    done = s.consumed_indices(consumed)
    assert done == sorted(full[:consumed])

    s2 = LengthGroupedLongestFirstSampler([5] * n, seed=7)
    s2.set_exclusion(done, epoch=0)
    assert len(s2) == n - consumed
    rest = list(iter(s2))
    assert rest == full[consumed:]  # the unconsumed examples keep their original relative order
    nxt = list(iter(s2))  # the following epoch is a full, fresh pass
    assert sorted(nxt) == list(range(n))
    # consumed_indices unions earlier-session exclusions with what this session trained on
    s3 = LengthGroupedLongestFirstSampler([5] * n, seed=7)
    s3.set_exclusion(done, epoch=0)
    order = list(iter(s3))
    assert set(s3.consumed_indices(3)) == set(done) | set(order[:3])


# ----------------------------------------------------------------------------------------------- tracking
def test_tracking_policy_matrix() -> None:
    kw = dict(is_main=True, has_key=False)
    assert decide_tracking(requested="auto", tuning_active=True, **kw).mode == "off"
    assert decide_tracking(requested="online", tuning_active=True, disable_during_tuning=False, is_main=True, has_key=True,
                           probe=lambda: True).mode == "online"
    assert decide_tracking(requested="auto", tuning_active=False, is_main=False, has_key=True).mode == "off"
    assert decide_tracking(requested="off", tuning_active=False, **kw).mode == "off"
    assert decide_tracking(requested="offline", tuning_active=False, **kw).mode == "offline"
    assert decide_tracking(requested="online", tuning_active=False, **kw).mode == "offline"  # no key
    assert decide_tracking(requested="auto", tuning_active=False, is_main=True, has_key=True, probe=lambda: False).mode == "offline"
    assert decide_tracking(requested="auto", tuning_active=False, is_main=True, has_key=True, probe=lambda: True).mode == "online"
    assert decide_tracking(requested="bogus", tuning_active=False, **kw).mode == "off"


def test_tracker_off_is_inert_and_offline_wandb_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = Tracker(requested="auto", tuning_active=True, is_main=True, run_dir=tmp_path, project="p", run_name="r")
    t.log(1, {"a": 1.0}); t.summary({"x": 1}); t.finish()  # all no-ops
    assert t.mode == "off" and not (tmp_path / "wandb").exists()

    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t2 = Tracker(requested="auto", tuning_active=False, is_main=True, run_dir=tmp_path, project="p", run_name="r")
    assert t2.mode in {"offline", "off"}  # offline normally; off only if this wandb build cannot start offline
    t2.log(1, {"loss": 1.0}); t2.summary({"final": 0.5}); t2.finish()


def test_tracker_survives_a_broken_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    boom = types.ModuleType("wandb")
    boom.init = lambda **_k: (_ for _ in ()).throw(RuntimeError("no network"))  # type: ignore[attr-defined]
    boom.Settings = lambda **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", boom)
    t = Tracker(requested="offline", tuning_active=False, is_main=True, run_dir=tmp_path, project="p", run_name="r")
    assert t.mode == "off" and "no network" in (t.failed or "")
    t.log(1, {"a": 1}); t.finish()


# ----------------------------------------------------------------------------------------------- secrets
def test_secret_scanner_flags_tokens_not_git_shas(tmp_path: Path) -> None:
    assert scan_text("token = hf_" + "a" * 34)[0].kind == "huggingface_token"
    assert scan_text('{"key": "' + "0" * 32 + '"}')[0].kind == "kaggle_json_key"
    assert scan_text("commit 4b9572c1d2e3f405a6b7c8d9e0f1a2b3c4d5e6f7 ok") == []
    assert scan_text("WANDB_API_KEY=" + "a" * 40)  # 40-hex next to 'wandb' is a key
    (tmp_path / "ok.json").write_text('{"loss": 0.1}')
    (tmp_path / ".env").write_text("X=1")
    (tmp_path / "leak.log").write_text("hf_" + "b" * 36)
    (tmp_path / "w.safetensors").write_bytes(b"hf_" + b"c" * 40)  # binary payloads are never scanned
    kinds = {(f.path, f.kind) for f in scan_dir(tmp_path)}
    assert (".env", "credential_file") in kinds and ("leak.log", "huggingface_token") in kinds
    assert not any(p == "w.safetensors" for p, _ in kinds)


# ----------------------------------------------------------------------------------------------- adapter contract
def _served_dir(path: Path, *, layers: int = 2, hidden: int = 16, inter: int = 32, packed: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    t: dict[str, np.ndarray] = {}
    for layer in range(layers):
        b = f"model.language_model.layers.{layer}"
        if packed:
            t[f"{b}.self_attn.q_proj.weight_packed"] = np.zeros((hidden, hidden // 8), dtype=np.int32)
            t[f"{b}.self_attn.q_proj.weight_shape"] = np.array([hidden, hidden], dtype=np.int64)
            t[f"{b}.mlp.up_proj.weight_packed"] = np.zeros((inter, hidden // 8), dtype=np.int32)
            t[f"{b}.mlp.up_proj.weight_shape"] = np.array([inter, hidden], dtype=np.int64)
        else:
            t[f"{b}.self_attn.q_proj.weight"] = np.zeros((hidden, hidden), dtype=np.float32)
            t[f"{b}.mlp.up_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
    t["model.vision_tower.layers.0.self_attn.q_proj.weight"] = np.zeros((4, 4), dtype=np.float32)
    save_file(t, str(path / "model.safetensors"))
    return path


@pytest.mark.parametrize("packed", [False, True])
def test_adapter_contract_passes_and_catches_shape_drift(tmp_path: Path, packed: bool) -> None:
    served = _served_dir(tmp_path / "served", packed=packed)
    ad = write_adapter(tmp_path / "ad")
    rep = check_adapter_contract(ad, served_model_dir=served, expected_rank=4)
    assert rep.ok, rep.errors
    assert check_adapter_contract(ad, served_model_dir=served, expected_rank=8).errors  # wrong rank
    wrong = _served_dir(tmp_path / "served2", hidden=24, packed=packed)
    bad = check_adapter_contract(ad, served_model_dir=wrong, expected_rank=4)
    assert not bad.ok and any("in_features" in e or "out_features" in e for e in bad.errors)


def test_adapter_contract_rejects_bad_keys_missing_module_and_dtype(tmp_path: Path) -> None:
    served = _served_dir(tmp_path / "served", layers=1)
    ad = write_adapter(tmp_path / "ad", layers=2)  # layer 1 does not exist in the served checkpoint
    errs = check_adapter_contract(ad, served_model_dir=served).errors
    assert any("layers.1" in e for e in errs)

    p = tmp_path / "bad"
    p.mkdir()
    save_file({"model.layers.0.q_proj.weight": np.zeros((2, 2), np.float32)}, str(p / "adapter_model.safetensors"))
    assert any("does not match" in e for e in check_adapter_contract(p).errors)
    p2 = tmp_path / "dt"
    p2.mkdir()
    base = "base_model.model.model.language_model.layers.0.mlp.up_proj"
    save_file({f"{base}.lora_A.weight": np.zeros((4, 16), np.float64), f"{base}.lora_B.weight": np.zeros((32, 4), np.float64)},
              str(p2 / "adapter_model.safetensors"))
    assert any("dtype" in e for e in check_adapter_contract(p2).errors)
    assert check_adapter_contract(tmp_path / "nothing").errors


# ----------------------------------------------------------------------------------------------- bundle
def _run_ckpts(tmp_path: Path) -> Path:
    root = tmp_path / "run" / "checkpoints"
    for ad in ("main_lora", "tool_lora"):
        write_checkpoint(root / ad, 10, fake_checkpoint_writer(10), keep=2)
        write_checkpoint(root / ad, 20, fake_checkpoint_writer(20), keep=2)
        write_best_adapter(root / ad, 15, 0.5, lambda d: write_adapter(d))
        write_adapter(root / ad / "final_adapter")
        ProgressLedger(ad).save(root / ad)
    return root


def test_sync_bundle_copies_latest_and_discovery_finds_it(tmp_path: Path) -> None:
    root = _run_ckpts(tmp_path)
    out = tmp_path / "work" / "lora_checkpoints"
    meta = sync_bundle(out, run_checkpoints_dir=root, adapters=["main_lora", "tool_lora", "ghost"],
                       served_model_id="gemma-4-e4b-it-qat-w4a16-ct", dataset_id="u/ckpts", run_id="r1")
    assert meta["adapters"]["main_lora"]["status"] == "ok" and meta["adapters"]["ghost"]["status"] == "empty"
    assert [p.name for p in list_checkpoints(out / "main_lora")] == ["step_000020"]
    assert (out / "main_lora" / "best_adapter" / "adapter_model.safetensors").is_file()
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == "u/ckpts"
    # a second sync with a newer checkpoint replaces (not accumulates) the resumable one
    write_checkpoint(root / "main_lora", 30, fake_checkpoint_writer(30), keep=5)
    sync_bundle(out, run_checkpoints_dir=root, adapters=["main_lora"], served_model_id="m", previous_sessions=1)
    assert [p.name for p in list_checkpoints(out / "main_lora")] == ["step_000030"]
    assert json.loads((out / "bundle.json").read_text())["sessions_total"] == 2

    inp = tmp_path / "kaggle_input"
    (inp / "ds").mkdir(parents=True)
    import shutil

    shutil.copytree(out, inp / "ds" / "ckpts")
    found = discover_bundles([inp])
    assert found == [inp / "ds" / "ckpts"]
    cands = resume_candidates(tmp_path / "empty", "main_lora", bundle_dirs=found)
    assert [c.name for c in cands] == ["step_000030"]
    assert discover_bundles([tmp_path / "does-not-exist"]) == []


def test_tuning_bundle_carries_only_study_files_and_survives_a_session_without_a_study(tmp_path: Path) -> None:
    study = tmp_path / "study"
    (study / "trials" / "trial_0000" / "checkpoints").mkdir(parents=True)
    (study / "trials" / "trial_0000" / "checkpoints" / "big.bin").write_bytes(b"x" * 1000)
    (study / "shared_cache").mkdir()
    (study / "shared_cache" / "train.parquet").write_bytes(b"p")
    (study / "study.db").write_bytes(b"sqlite")
    (study / "best_params.json").write_text('{"params": {"train.lr": 0.0001}, "sets": ["train.lr=0.0001"]}')
    out1 = tmp_path / "bundle1"
    sync_bundle(out1, run_checkpoints_dir=tmp_path / "none", adapters=[], served_model_id="m", tuning_dir=study)
    assert sorted(p.name for p in (out1 / "tuning").iterdir()) == ["best_params.json", "study.db"]  # no trials / caches
    # A later normal run has no study of its own; it must still hand the attached bundle's results on to the next bundle.
    assert tuning_source(None, [out1]) == out1 / "tuning"
    assert tuning_source(study, [out1]) == study
    out2 = tmp_path / "bundle2"
    sync_bundle(out2, run_checkpoints_dir=tmp_path / "none", adapters=[], served_model_id="m", tuning_dir=tuning_source(None, [out1]))
    assert (out2 / "tuning" / "best_params.json").is_file() and (out2 / "tuning" / "study.db").is_file()
    assert best_params_from_bundles([out2])
    # Syncing a bundle onto itself must not fail or truncate the files.
    sync_bundle(out2, run_checkpoints_dir=tmp_path / "none", adapters=[], served_model_id="m", tuning_dir=out2 / "tuning")
    assert (out2 / "tuning" / "study.db").read_bytes() == b"sqlite"
    assert tuning_source(None, [tmp_path / "nothing"]) is None


def test_run_applies_tuned_params_from_an_attached_bundle(tmp_path: Path) -> None:
    from pipeline.multi_lora_sft.run import with_tuned_params
    from pipeline.multi_lora_sft.shared.fsio import atomic_write_json

    bundle = tmp_path / "bundle"
    atomic_write_json(bundle / "bundle.json", {"schema": 1})
    atomic_write_json(bundle / "tuning" / "best_params.json",
                      {"trial": 3, "value": 1.5, "params": {"train.lr": 2e-4}, "sets": ["train.lr=0.0002", "lora.r=16", "lora.alpha=32"]})
    base = [f"resume.bundle_dirs=[{bundle.as_posix()!r}]".replace("'", '"'), "resume.auto_discover=false"]
    cfg, sets, note = with_tuned_params(None, base)
    assert cfg.train.lr == 2e-4 and cfg.lora.r == 16 and cfg.lora.alpha == 32
    assert note and "trial 3" in note
    # explicit overrides still win over the tuned value
    cfg2, _, _ = with_tuned_params(None, [*base, "train.lr=5e-5"])
    assert cfg2.train.lr == 5e-5
    # never inside a tuning trial, and never when switched off
    assert with_tuned_params(None, [*base, "tuning.active=true"])[2] is None
    assert with_tuned_params(None, [*base, "tuning.use_best_params=false"])[2] is None
    # a bundle that already has checkpoints keeps its LoRA shape: tuned lora.* is dropped, the rest applies
    from pipeline.multi_lora_sft.shared.checkpoints import write_checkpoint

    write_checkpoint(bundle / "main_lora", 5, fake_checkpoint_writer(5))
    cfg3, _, note3 = with_tuned_params(None, base)
    assert note3 and "lora.* ignored" in note3 and cfg3.train.lr == 2e-4 and cfg3.lora.r != 16


def test_bundle_removes_files_that_contain_secrets(tmp_path: Path) -> None:
    root = _run_ckpts(tmp_path)
    (root / "main_lora" / "step_000020" / "notes.json").write_text('{"token": "hf_' + "z" * 34 + '"}')
    out = tmp_path / "b"
    meta = sync_bundle(out, run_checkpoints_dir=root, adapters=["main_lora"], served_model_id="m")
    assert meta["secrets_removed"] and not (out / "main_lora" / "step_000020" / "notes.json").exists()
    assert (root / "main_lora" / "step_000020" / "notes.json").exists()  # the original is untouched


def test_resume_candidates_order_run_then_explicit_then_bundle(tmp_path: Path) -> None:
    run = tmp_path / "run_ckpts"
    write_checkpoint(run / "a", 5, fake_checkpoint_writer(5))
    prev = tmp_path / "prev"
    write_checkpoint(prev / "checkpoints" / "a", 9, fake_checkpoint_writer(9))
    bundle = tmp_path / "bundle"
    write_checkpoint(bundle / "a", 99, fake_checkpoint_writer(99))
    names = [str(c.relative_to(tmp_path)).replace("\\", "/") for c in resume_candidates(run, "a", resume_from=str(prev), bundle_dirs=[bundle])]
    assert names == ["run_ckpts/a/step_000005", "prev/checkpoints/a/step_000009", "bundle/a/step_000099"]


def test_best_params_from_bundle(tmp_path: Path) -> None:
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning" / "best_params.json").write_text(json.dumps({"params": {"train.lr": 1e-4}, "sets": ["train.lr=0.0001"], "trial": 3, "value": 0.7}))
    assert best_params_from_bundles([tmp_path])["trial"] == 3
    sets, note = apply_best_params(["x=1"], [tmp_path])
    assert sets == ["train.lr=0.0001", "x=1"] and "trial 3" in note
    assert apply_best_params(["x=1"], [tmp_path / "none"]) == (["x=1"], None)


# ----------------------------------------------------------------------------------------------- backend choice
@pytest.mark.parametrize(
    "requested, model, acc, tpu, expect",
    [
        ("auto", "gemma-4-e4b-it-qat-w4a16-ct", "auto", True, "tunix"),
        ("auto", "gemma-4-e2b-it-qat-w4a16-ct", "tpu", False, "tunix"),
        ("auto", "gemma-4-e4b-it-qat-w4a16-ct", "auto", False, "trl"),
        ("auto", "gemma-4-31b-it-qat-w4a16-ct", "auto", True, "trl"),
        ("auto", "gemma-4-e4b-it-qat-w4a16-ct", "cuda", True, "trl"),
        ("trl", "gemma-4-e4b-it-qat-w4a16-ct", "tpu", True, "trl"),
        ("tunix", "gemma-4-31b-it-qat-w4a16-ct", "auto", False, "tunix"),
    ],
)
def test_resolve_backend(requested, model, acc, tpu, expect) -> None:
    assert resolve_backend(requested, served_model_id=model, accelerator=acc, tpu=tpu) == expect


# ----------------------------------------------------------------------------------------------- config rules
def test_float_overrides_in_exponent_notation_without_a_dot_are_numbers() -> None:
    # PyYAML parses `5e-5` / `1e-05` as strings; the notebook's json.dumps(1e-05) produces exactly that.
    cfg = load_config(None, ["train.lr=5e-5", "train.max_grad_norm=1e-05", "schedule.final_lr_ratio=1e-1"])
    assert (cfg.train.lr, cfg.train.max_grad_norm, cfg.schedule.final_lr_ratio) == (5e-5, 1e-5, 0.1)
    with pytest.raises(ConfigError):
        load_config(None, ["train.lr=fast"])


def test_wandb_off_survives_yaml_boolean_parsing() -> None:
    # `--set tracking.wandb=off` is parsed by YAML as the boolean False; it must still mean "off".
    assert load_config(None, ["tracking.wandb=off"]).tracking.wandb == "off"
    assert load_config(None, ["tracking.wandb=on"]).tracking.wandb == "auto"
    assert load_config(None, ["tracking.wandb=offline"]).tracking.wandb == "offline"


def test_new_config_sections_validate() -> None:
    cfg = load_config(None, ["train.backend=tunix", "train.accelerator=tpu", "train.parallelism=spmd", "model.quantization=bf16"])
    assert cfg.train.backend == "tunix"
    for bad in (
        ["train.backend=magic"],
        ["train.backend=tunix", "train.accelerator=cuda", "train.parallelism=single"],
        ["tracking.wandb=sometimes"],
        ["tuning.sampler=grid"],
        ["tuning.trials=0"],
        ["tuning.adapter=ghost"],
        ["output.dataset_id=not-a-slug"],
        ["resume.on_data_change=whatever"],
        ["train.loss_spike_factor=-1"],
    ):
        with pytest.raises(ConfigError):
            load_config(None, bad)
    assert load_config(None, ['tuning.space={"train.lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-3, "log": true}}']).tuning.space


# ----------------------------------------------------------------------------------------------- tuning helpers
def test_space_validation_and_param_sets() -> None:
    assert set(validate_space(DEFAULT_SPACE)) == set(DEFAULT_SPACE)
    s = validate_space({"train.lr": {"type": "float", "low": "1e-5", "high": "1e-3", "log": True}})  # YAML gives strings for 1e-5
    assert s["train.lr"]["low"] == pytest.approx(1e-5)
    for bad in ({"x": {"type": "weird"}}, {"x": {"type": "float", "low": 2, "high": 1}}, {"x": {"type": "float", "low": 0, "high": 1, "log": True}},
                {"x": {"type": "categorical", "choices": []}}):
        with pytest.raises(TuningError):
            validate_space(bad)
    sets = params_to_sets({"train.lr": 1e-4, "lora.r": 64, "lora.alpha_per_r": 2}, base_r=32)
    assert "lora.alpha=128" in sets and "lora.r=64" in sets and not any("alpha_per_r" in x for x in sets)
    assert "lora.alpha=64" in params_to_sets({"lora.alpha_per_r": 2}, base_r=32)


def test_tuned_values_beat_per_adapter_overrides_and_trial_sets_are_quiet() -> None:
    ov = {"lora": {"r": 16, "alpha": 32, "dropout": 0.1}, "train": {"max_seq_length": 4096, "lr": 5e-5}}
    out = strip_tuned_overrides(ov, ["lora.r", "lora.alpha_per_r", "train.lr"])
    assert out == {"lora": {"dropout": 0.1}, "train": {"max_seq_length": 4096}}

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "e4b_tpu.yaml", ["tuning.trial_minutes=7", "tuning.trial_max_steps=30"])
    sets = build_trial_sets(cfg, "tool_lora", ["train.lr=0.0001"], study_dir=Path("/tmp/study"), trial_no=3, shared_cache=None)
    joined = "\n".join(sets)
    for must in ('run.run_id="trial_0003"', "tuning.active=true", "eval.enabled=false", "output.bundle_dir=null",
                 'tracking.wandb="off"', "run.calibration_minutes=7.0", "train.max_steps=30", "train.lr=0.0001"):
        assert must in joined, must
    # the sets must load back into a valid config
    trial_cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "e4b_tpu.yaml", sets)
    assert trial_cfg.tuning.active and [a.name for a in trial_cfg.adapters] == ["tool_lora"] and trial_cfg.tracking.wandb == "off"
    assert trial_cfg.adapters[0].overrides.get("lora") == {"r": 16, "alpha": 32}  # untuned overrides survive


def test_trial_objective_prefers_validation_loss(tmp_path: Path) -> None:
    run = tmp_path / "t"
    (run / "metrics").mkdir(parents=True)
    (run / "metrics" / "a.jsonl").write_text("\n".join(json.dumps({"step": i, "train_loss": 2.0 - i * 0.1}) for i in range(8)) + "\n{half")
    v, src = trial_objective(run, "a")
    assert src == "train_loss_tail" and v == pytest.approx(sum(2.0 - i * 0.1 for i in range(3, 8)) / 5)
    (run / "manifest.json").write_text(json.dumps({"training_summaries": {"a": {"final_val_loss": 0.8}}}))
    assert trial_objective(run, "a") == (0.8, "val_loss")
    assert trial_objective(tmp_path / "none", "a") == (None, "none")


def test_run_study_with_fake_runner_resumes_and_writes_best_params(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from pipeline.multi_lora_sft.shared.tuning import run_study

    store = tmp_path / "study"
    calls: list[int] = []

    def fake_runner(cmd, trial_dir, trial, adapter, poll):
        calls.append(trial.number)
        assert "--set" in cmd and any(c == "tuning.active=true" for c in cmd)
        lr = trial.params["train.lr"]
        (trial_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (trial_dir / "cache" / adapter).mkdir(parents=True, exist_ok=True)
        (trial_dir / "manifest.json").write_text(json.dumps({"training_summaries": {adapter: {"final_val_loss": abs(lr - 1e-4) * 1000 + 0.5}}}))
        return 0

    base = [f"tuning.storage_dir={json.dumps(str(store))}", "tuning.trials=4", "tuning.n_startup_trials=2", "tuning.trial_minutes=1",
            'tuning.space={"train.lr": {"type": "float", "low": 3.0e-5, "high": 3.0e-4, "log": true}}', "budget.session_hours=2"]
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "e4b_tpu.yaml"
    res = run_study(cfg_path, base, runner=fake_runner)
    assert res.completed == 4 and res.best_params and len(calls) == 4
    best = json.loads((store / "best_params.json").read_text())
    assert best["value"] == pytest.approx(res.best_value) and best["sets"][0].startswith("train.lr=")
    assert (store / "shared_cache").is_dir() and (store / "trials.json").is_file()
    # a second invocation resumes the same study instead of starting over
    res2 = run_study(cfg_path, [*base[:1], "tuning.trials=6", *base[2:]], runner=fake_runner)
    assert res2.completed == 6 and len(calls) == 6


def test_run_study_survives_failed_trials(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from pipeline.multi_lora_sft.shared.tuning import run_study

    n = {"i": 0}

    def flaky(cmd, trial_dir, trial, adapter, poll):
        n["i"] += 1
        trial.suggest_float("dummy", 0, 1)
        if n["i"] == 1:
            return 1  # crashed before writing anything
        (trial_dir).mkdir(parents=True, exist_ok=True)
        (trial_dir / "manifest.json").write_text(json.dumps({"training_summaries": {adapter: {"final_val_loss": 1.0}}}))
        return 0

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "e4b_tpu.yaml"
    res = run_study(cfg_path, [f"tuning.storage_dir={json.dumps(str(tmp_path / 's'))}", "tuning.trials=3", "tuning.trial_minutes=1",
                               'tuning.space={"train.lr": {"type": "float", "low": 3.0e-5, "high": 3.0e-4, "log": true}}',
                               "budget.session_hours=2"], runner=flaky)
    assert res.failed == 1 and res.completed >= 1 and res.best_params is not None
