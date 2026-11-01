"""`train.parallelism=fsdp` on CPU: two real processes (gloo) running FSDP1 over the tiny Gemma 4 + LoRA.

The 4xL4 path cannot run here (no CUDA, no bitsandbytes kernels), but everything FSDP-specific that the pipeline adds
does, on the same torch/peft/transformers code:
- the selective loss reads the captured hidden states and `lm_head` outside the root forward; without the zero-weight
  logits anchor in `compute_loss` the root's end-of-backward callback never runs and the second step fails;
- gradients through the sharded model match an unsharded reference (same batch on both ranks, so FSDP's averaging
  must reproduce the single-process gradient exactly);
- `gather_lora_state_dict` returns full LoRA tensors under the PeftModel's own key names;
- the name-keyed optimizer state round-trips through `fsdp_full_optim_state_dict` / `fsdp_optim_state_to_load`;
- greedy parity decoding runs in lockstep through the FSDP root, with and without the adapter;
- `SyncedClock` / `agree_any` give every rank the same decision;
- bf16 mixed precision with `buffer_dtype=None` leaves the fp32 RoPE buffers alone.
"""

from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

WORLD = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _fake_trainer(world: int) -> Any:
    from pipeline.multi_lora_sft.training.loss import build_selective_sft_trainer_cls

    cls = build_selective_sft_trainer_cls()
    t = object.__new__(cls)  # only compute_loss is exercised; it needs these attributes and nothing from __init__
    t.loss_impl = "selective"
    t.chunk_rows = 7  # several chunks per batch
    t.checkpoint_chunks = True
    t._softcap = 30.0
    t._tuple_outputs = False
    t.loss_is_scaled_for_ga = True
    t.args = SimpleNamespace(world_size=world, average_tokens_across_devices=True)
    t.accelerator = SimpleNamespace(num_processes=world)
    return t


def _batch(vocab: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(5, vocab - 10, (2, 24), generator=g)
    labels = ids.clone()
    labels[:, :9] = -100  # prompt part unsupervised
    attn = torch.ones_like(ids)
    weights = torch.ones(ids.shape, dtype=torch.float32)
    weights[:, 15:18] = 1.5
    return {"input_ids": ids, "attention_mask": attn, "labels": labels, "weights": weights}


def _weight_sum(batch: dict[str, torch.Tensor]) -> float:
    mask = batch["labels"][:, 1:].ne(-100).float()
    return float((batch["weights"][:, 1:] * mask).sum())


def _load(model_dir: str, dtype: torch.dtype) -> Any:
    from transformers import Gemma4ForConditionalGeneration

    from pipeline.multi_lora_sft.config import LoraSection
    from pipeline.multi_lora_sft.training.model import attach_lora_adapter

    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1)  # same LoRA init on every rank (as the train phase does)
    # init_lora_weights=False: random A and B, so both receive gradients from the first step
    peft_model, _, _ = attach_lora_adapter(model, LoraSection(r=4, alpha=8, init_lora_weights=False))
    return peft_model


def _wrap(peft_model: Any, mixed: Any = None) -> Any:
    from peft.utils.other import fsdp_auto_wrap_policy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "Gemma4TextDecoderLayer"
    return FSDP(
        peft_model,
        auto_wrap_policy=fsdp_auto_wrap_policy(peft_model),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=False,
        sync_module_states=False,
        mixed_precision=mixed,
        device_id=torch.device("cpu"),
    )


def _worker(rank: int, port: int, model_dir: str, out_dir: str) -> None:
    import torch.distributed as dist

    from pipeline.multi_lora_sft.training.distributed import (
        SyncedClock,
        agree_all,
        agree_any,
        fsdp_full_optim_state_dict,
        fsdp_optim_state_to_load,
        gather_lora_state_dict,
        is_fsdp_wrapped,
    )
    from pipeline.multi_lora_sft.training.parity import greedy_decode_with_logprobs

    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(WORLD), "LOCAL_RANK": str(rank)})
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    res: dict[str, Any] = {}
    try:
        # ---------------------------------------------------------------- rank agreement helpers
        clock = SyncedClock(now_fn=lambda: 1000.0 * (rank + 1))
        res["clock"] = clock.sync()
        res["agree_any"] = agree_any(rank == 1)
        res["agree_all"] = agree_all(rank == 1)

        # ---------------------------------------------------------------- fp32: exact equivalence with a reference
        peft_model = _load(model_dir, torch.float32)
        ref = copy.deepcopy(peft_model) if rank == 0 else None
        model = _wrap(peft_model)
        res["is_fsdp"] = is_fsdp_wrapped(model)
        batch = _batch(int(model.module.config.get_text_config().vocab_size))
        wsum = _weight_sum(batch)

        trainer = _fake_trainer(WORLD)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        ref_opt = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr=0.05) if ref is not None else None
        ref_trainer = _fake_trainer(1)
        losses, ref_losses = [], []
        for _ in range(3):  # >1 step: the root's end-of-backward state reset must have run after step 1
            opt.zero_grad(set_to_none=True)
            # same batch on both ranks; the step-wide weight sum covers both ranks' copies
            loss = trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(WORLD * wsum))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if ref is not None:
                ref_opt.zero_grad(set_to_none=True)
                rl = ref_trainer.compute_loss(ref, dict(batch), num_items_in_batch=torch.tensor(wsum))
                rl.backward()
                ref_opt.step()
                ref_losses.append(float(rl))
        res["losses"] = losses

        gathered = gather_lora_state_dict(model)
        if rank == 0:
            ref_sd = {k: v for k, v in ref.state_dict().items() if "lora_" in k}
            res["ref_losses"] = ref_losses
            res["gathered_keys_match"] = sorted(gathered) == sorted(ref_sd)
            res["max_weight_diff"] = max(float((gathered[k] - ref_sd[k]).abs().max()) for k in ref_sd)
            res["n_lora_tensors"] = len(gathered)
        else:
            res["gathered_on_other_rank"] = len(gathered)

        # ---------------------------------------------------------------- optimizer state round trip
        adam = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        adam.zero_grad(set_to_none=True)
        trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(WORLD * wsum)).backward()
        adam.step()
        full = fsdp_full_optim_state_dict(model, adam)
        osd_path = Path(out_dir) / "optim.pt"
        if rank == 0:
            torch.save(full, osd_path)
            res["optim_state_entries"] = len(full["state"])
        dist.barrier()
        loaded_full = torch.load(osd_path, map_location="cpu", weights_only=False)
        adam2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        adam2.load_state_dict(fsdp_optim_state_to_load(model, adam2, loaded_full))
        diffs = []
        for p in (q for q in model.parameters() if q.requires_grad):
            for key in ("exp_avg", "exp_avg_sq"):
                diffs.append(float((adam.state[p][key] - adam2.state[p][key]).abs().max()))
        res["optim_roundtrip_max_diff"] = max(diffs)

        # ---------------------------------------------------------------- adapter save + reload + checkpoint + best
        # Regression for Bug 1: PEFT 0.21 filters `state_dict` against `model.named_modules()`, which under FSDP1
        # contain `_fsdp_wrapped_module.` and matched 0 keys. Verify all three save paths write all LoRA tensors and
        # reload identically onto a fresh unsharded base model + pass export validation.
        gathered = gather_lora_state_dict(model)
        if rank == 0:
            from peft import PeftModel, set_peft_model_state_dict
            from safetensors.torch import load_file as load_safetensors
            from transformers import Gemma4ForConditionalGeneration

            from pipeline.multi_lora_sft.phases.export import export_single_adapter
            from pipeline.multi_lora_sft.shared.checkpoints import (
                checkpoint_is_loadable,
                read_best_adapter,
                write_best_adapter,
            )
            from pipeline.multi_lora_sft.training.callbacks import (
                save_adapter_weights,
                save_wallclock_checkpoint,
            )
            from pipeline.multi_lora_sft.training.schedule import WSDScheduleController

            ref.load_state_dict(gathered, strict=False)

            saved_dir = Path(out_dir) / "saved_final_adapter"
            save_adapter_weights(model, saved_dir, state_dict=gathered)
            saved_st = load_safetensors(str(saved_dir / "adapter_model.safetensors"), device="cpu")
            res["saved_tensor_count"] = len(saved_st)

            fresh_base = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
            fresh_base.eval()
            reloaded = PeftModel.from_pretrained(fresh_base, str(saved_dir))
            reloaded.eval()
            ref.eval()
            with torch.no_grad():
                ref_logits = ref(input_ids=batch["input_ids"]).logits
                rel_logits = reloaded(input_ids=batch["input_ids"]).logits
            res["reloaded_max_logit_diff"] = float((ref_logits - rel_logits).abs().max())

            # Resume path (`set_peft_model_state_dict`) must match every key with 0 unexpected keys
            resume_target = _load(model_dir, torch.float32)
            incompat = set_peft_model_state_dict(resume_target, saved_st)
            res["resume_unexpected_keys"] = list(getattr(incompat, "unexpected_keys", []) or [])

            # Export validation (`export_single_adapter`) must accept the saved adapter directory
            export_meta = export_single_adapter(
                adapter_name="main_lora",
                source_adapter_dir=saved_dir,
                output_adapter_dir=Path(out_dir) / "exported_main_lora",
                served_model_id="gemma-4-31b-it-qat-w4a16-ct",
            )
            res["exported_module_count"] = int(export_meta["module_count"])

            # Wall-clock checkpoint + best_adapter paths under FSDP
            wsd_tmp = WSDScheduleController(total_epoch_steps=10, t_start=0.0, deadline=100.0)
            ckpt_dir = save_wallclock_checkpoint(
                checkpoint_root=Path(out_dir) / "ckpts",
                step=4,
                epoch=0.5,
                model=model,
                optimizer=adam,
                scheduler=None,
                wsd_controller=wsd_tmp,
                sampler=None,
                samples_in_epoch=0,
                keep_checkpoints=2,
                is_main_process=True,
                adapter_state=gathered,
                optimizer_state=full,
            )
            res["wallclock_ckpt_loadable"] = bool(ckpt_dir is not None and checkpoint_is_loadable(ckpt_dir))
            write_best_adapter(
                Path(out_dir) / "ckpts",
                step=4,
                val_loss=1.25,
                copy_fn=lambda d, m=model, g=gathered: save_adapter_weights(m, d, state_dict=g),
            )
            res["best_adapter_loadable"] = read_best_adapter(Path(out_dir) / "ckpts") is not None
            del fresh_base, reloaded, resume_target

        # ---------------------------------------------------------------- lockstep greedy parity decode + budget
        from pipeline.multi_lora_sft.training.parity import generate_hf_parity_reference

        prompt = batch["input_ids"][0, :10].tolist()
        tok = SimpleNamespace(eos_token_id=None, decode=lambda ids, skip_special_tokens=False: str(ids))
        with_adapter = greedy_decode_with_logprobs(model, tok, prompt, gen_tokens=5)
        with model.module.disable_adapter():
            base = greedy_decode_with_logprobs(model, tok, prompt, gen_tokens=5)
        res["parity_tokens"] = [s["token_id"] for s in with_adapter]
        res["parity_base_tokens"] = [s["token_id"] for s in base]

        # Rank-agreed stop/deadline on FSDP parity: rank 1 signalling stop halts all ranks cleanly without deadlock
        stopped_steps = greedy_decode_with_logprobs(
            model, tok, prompt, gen_tokens=5, stop_requested=lambda: rank == 1
        )
        res["parity_stopped_len"] = len(stopped_steps)
        parity_recs = [{"task_id": "t0", "input_ids": batch["input_ids"][0].tolist(), "labels": batch["labels"][0].tolist()}]
        expired_out = generate_hf_parity_reference(
            adapter_name="main_lora",
            peft_model=model,
            tokenizer=tok,
            val_records=parity_recs,
            train_records=parity_recs,
            output_path=Path(out_dir) / "expired_parity.json",
            num_prompts=2,
            gen_tokens=4,
            write=(rank == 0),
            deadline=0.0,
        )
        res["parity_expired_none"] = expired_out is None
        if rank == 0:
            res["ref_parity_tokens"] = [s["token_id"] for s in greedy_decode_with_logprobs(ref, tok, prompt, gen_tokens=5)]
            with ref.disable_adapter():
                res["ref_parity_base_tokens"] = [
                    s["token_id"] for s in greedy_decode_with_logprobs(ref, tok, prompt, gen_tokens=5)
                ]
        del model, peft_model, ref

        # ---------------------------------------------------------------- bf16 base + bf16 mixed precision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        bf = _load(model_dir, torch.bfloat16)
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=None)
        bf_model = _wrap(bf, mixed=mp)
        for unit in FSDP.fsdp_modules(bf_model):  # what accelerate does: trainable flat params kept in fp32
            flat = getattr(unit, "_flat_param", None)
            if flat is not None and flat.requires_grad and flat.dtype != torch.float32:
                flat.data = flat.data.to(torch.float32)
                unit._handle._orig_param_dtype = torch.float32
        adam_bf = torch.optim.AdamW([p for p in bf_model.parameters() if p.requires_grad], lr=1e-3)
        bf_losses = []
        for _ in range(2):
            adam_bf.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):  # the Trainer's bf16=True autocast around compute_loss
                loss = trainer.compute_loss(bf_model, dict(batch), num_items_in_batch=torch.tensor(WORLD * wsum))
            loss.backward()
            adam_bf.step()
            bf_losses.append(float(loss.detach()))
        res["bf16_losses"] = bf_losses
        res["inv_freq_dtypes"] = sorted(
            {str(b.dtype) for n, b in bf_model.named_buffers() if n.endswith("inv_freq")}
        )
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(res), encoding="utf-8")
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def fsdp_results(tiny_gemma4_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with gloo is not available")
    out = tmp_path_factory.mktemp("fsdp_cpu")
    mp.spawn(_worker, args=(_free_port(), str(tiny_gemma4_dir), str(out)), nprocs=WORLD, join=True)
    return [json.loads((out / f"rank{r}.json").read_text(encoding="utf-8")) for r in range(WORLD)]


def test_rank_agreement_helpers(fsdp_results: list[dict[str, Any]]) -> None:
    assert [r["clock"] for r in fsdp_results] == [1000.0, 1000.0]  # rank 0's clock everywhere
    assert [r["agree_any"] for r in fsdp_results] == [True, True]
    assert [r["agree_all"] for r in fsdp_results] == [False, False]


def test_fsdp_training_matches_unsharded_reference(fsdp_results: list[dict[str, Any]]) -> None:
    r0, r1 = fsdp_results
    assert r0["is_fsdp"] and r1["is_fsdp"]
    assert r0["losses"] == pytest.approx(r1["losses"], rel=1e-6)
    assert r0["losses"] == pytest.approx(r0["ref_losses"], rel=1e-5)
    assert r0["losses"][2] < r0["losses"][0]  # it actually trains
    assert r0["gathered_keys_match"] and r0["n_lora_tensors"] > 0
    assert r0["max_weight_diff"] < 1e-5
    assert r1["gathered_on_other_rank"] == 0  # rank 0 only


def test_fsdp_optimizer_state_round_trip(fsdp_results: list[dict[str, Any]]) -> None:
    assert fsdp_results[0]["optim_state_entries"] == fsdp_results[0]["n_lora_tensors"]
    assert all(r["optim_roundtrip_max_diff"] == 0.0 for r in fsdp_results)


def test_fsdp_lockstep_parity_decode(fsdp_results: list[dict[str, Any]]) -> None:
    r0, r1 = fsdp_results
    assert r0["parity_tokens"] == r1["parity_tokens"] == r0["ref_parity_tokens"]
    assert r0["parity_base_tokens"] == r1["parity_base_tokens"] == r0["ref_parity_base_tokens"]
    assert [r["parity_stopped_len"] for r in fsdp_results] == [0, 0]
    assert [r["parity_expired_none"] for r in fsdp_results] == [True, True]


def test_fsdp_saved_adapter_reloads_and_exports_nonempty(fsdp_results: list[dict[str, Any]]) -> None:
    r0 = fsdp_results[0]
    assert r0["saved_tensor_count"] == r0["n_lora_tensors"] > 0
    assert r0["reloaded_max_logit_diff"] < 1e-5
    assert r0["resume_unexpected_keys"] == []
    assert r0["exported_module_count"] == r0["n_lora_tensors"] // 2
    assert r0["wallclock_ckpt_loadable"] is True
    assert r0["best_adapter_loadable"] is True


def test_fsdp_bf16_mixed_precision_keeps_rope_buffers(fsdp_results: list[dict[str, Any]]) -> None:
    for r in fsdp_results:
        assert all(torch.isfinite(torch.tensor(r["bf16_losses"])))
        assert r["inv_freq_dtypes"] == ["torch.float32"]


def test_keep_buffers_in_native_dtype_survives_later_set_mixed_precision() -> None:
    from accelerate.utils import FullyShardedDataParallelPlugin

    from pipeline.multi_lora_sft.training.distributed import (
        keep_buffers_in_native_dtype,
    )

    plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sync_module_states=False, mixed_precision_policy="bf16")
    assert plugin.mixed_precision_policy.buffer_dtype == torch.bfloat16  # accelerate's default casts buffers
    keep_buffers_in_native_dtype(plugin)
    assert plugin.mixed_precision_policy.buffer_dtype is None
    plugin.set_mixed_precision(torch.bfloat16, override=True)  # what the Trainer does for QLoRA before wrapping
    assert plugin.mixed_precision_policy.buffer_dtype is None
    assert plugin.mixed_precision_policy.param_dtype == torch.bfloat16


def test_collective_device_handles_hybrid_fsdp_backend_and_env_precedes_init(monkeypatch: pytest.MonkeyPatch) -> None:
    import ast
    import inspect

    import torch.distributed as dist

    from pipeline.multi_lora_sft.phases import train as train_phase
    from pipeline.multi_lora_sft.training import distributed as dist_mod

    # Hybrid `cuda:nccl,cpu:gloo` backend selected by accelerate PartialState when FSDP + FULL_STATE_DICT is active.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "cuda:nccl,cpu:gloo")
    assert dist_mod._collective_device() == torch.device("cuda", 2)

    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    assert dist_mod._collective_device() == torch.device("cpu")

    # `apply_fsdp_env()` call must precede `init_distributed()` call in `run_train_adapter_phase` so PartialState sees FSDP_ENV.
    tree = ast.parse(inspect.getsource(train_phase.run_train_adapter_phase))
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"apply_fsdp_env", "init_distributed"}
    }
    assert call_lines["apply_fsdp_env"] < call_lines["init_distributed"]


def test_save_adapter_weights_and_best_adapter_fail_on_zero_tensors(
    tiny_gemma4_dir: Path, tmp_path: Path
) -> None:
    from safetensors.torch import save_file as save_safetensors

    from pipeline.multi_lora_sft.shared.checkpoints import (
        CheckpointError,
        write_best_adapter,
    )
    from pipeline.multi_lora_sft.training.callbacks import save_adapter_weights

    peft_model = _load(str(tiny_gemma4_dir), torch.float32)
    with pytest.raises(CheckpointError, match="0 LoRA tensors"):
        save_adapter_weights(peft_model, tmp_path / "empty_adapter", state_dict={})

    with pytest.raises(CheckpointError, match="0 tensors"):
        write_best_adapter(
            tmp_path / "ckpts",
            step=1,
            val_loss=1.0,
            copy_fn=lambda d: save_safetensors({}, str(d / "adapter_model.safetensors")),
        )


def test_callback_resyncs_stale_clock_after_model_load(tmp_path: Path) -> None:
    from transformers import TrainerControl, TrainerState, TrainingArguments

    from pipeline.multi_lora_sft.training.callbacks import MultiLoraTrainingCallback
    from pipeline.multi_lora_sft.training.distributed import SyncedClock
    from pipeline.multi_lora_sft.training.schedule import WSDScheduleController

    fake_now = [1000.0]
    clock = SyncedClock(now_fn=lambda: fake_now[0])
    # Simulate distributed mode where `clock()` caches `_value` from the last `sync()`.
    clock._value = 1000.0
    wsd = WSDScheduleController(
        warmup_steps=2,
        decay_frac=0.15,
        final_lr_ratio=0.1,
        total_epoch_steps=100,
        t_start=1000.0,
        deadline=3000.0,
        now_fn=clock,
    )
    tracker = {"step_total_tokens": 2048.0, "step_supervised_tokens": 1024.0}
    cb = MultiLoraTrainingCallback(
        adapter_name="main_lora",
        wsd_controller=wsd,
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
        checkpoint_root=tmp_path / "ckpts",
        metrics_tracker=tracker,
        sampler=None,
        dataset_total_tokens=100_000,
        samples_per_step=4,
        log_every_steps=1,
        val_every_min=5.0,
        checkpoint_every_min=5.0,
        has_eval_dataset=True,
        clock=clock,
    )

    # 7-minute model load + FSDP wrapping happens before `on_train_begin`
    fake_now[0] = 1420.0
    args = TrainingArguments(output_dir=str(tmp_path / "hf"), per_device_train_batch_size=1, report_to="none")
    state = TrainerState(global_step=0, epoch=0.0, is_world_process_zero=True)
    control = TrainerControl()
    cb.on_train_begin(args, state, control)
    assert wsd.t_start == 1420.0
    assert cb._last_ckpt_ts == 1420.0
    assert cb._last_val_ts == 1420.0
    assert cb._window_start_ts == 1420.0

    # Step 1 takes 20s -> 2048 tok / 20s = 102.4 tok/s (NOT 2048 / 440s = 4.65 tok/s) and does not trigger 5m val/ckpt!
    fake_now[0] = 1440.0
    state.global_step = 1
    control = cb.on_step_end(args, state, TrainerControl())
    assert not control.should_evaluate
    cb.on_log(args, state, control, logs={"loss": 1.5, "grad_norm": 0.5, "learning_rate": 1e-4})
    logged = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["tokens_per_sec"] == pytest.approx(102.4, rel=1e-3)


def test_gather_peak_memory_mb_and_report_list_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch.distributed as dist

    from pipeline.multi_lora_sft.config import PipelineConfig
    from pipeline.multi_lora_sft.phases.report import training_health_lines
    from pipeline.multi_lora_sft.runctx import RunContext
    from pipeline.multi_lora_sft.training import distributed as dist_mod

    monkeypatch.setattr(dist_mod, "dist_ready", lambda: True)
    monkeypatch.setattr(dist_mod, "world_size", lambda: 4)
    monkeypatch.setattr(dist_mod, "rank", lambda: 2)
    monkeypatch.setattr(dist_mod, "_collective_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda idx: (18000 * 1024 * 1024) if idx == 2 else 0)

    def _fake_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        # Simulate the other 3 ranks contributing their own peak allocations into MAX all-reduce
        other = torch.tensor([18034.2, 17990.5, 0.0, 18012.8], dtype=tensor.dtype)
        tensor.copy_(torch.maximum(tensor, other))

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)
    peaks = dist_mod.gather_peak_memory_mb("cuda")
    assert peaks == [18034.2, 17990.5, 18000.0, 18012.8]

    ctx = RunContext(tmp_path / "run")
    ctx.ensure_dirs()
    (ctx.metrics_dir / "main_lora.jsonl").write_text(
        json.dumps({"step": 1, "train_loss": 2.0, "grad_norm": 1.0, "tokens_per_sec": 340.0, "peak_gpu_mem_mb": peaks})
        + "\n",
        encoding="utf-8",
    )
    lines = "\n".join(training_health_lines(ctx, PipelineConfig(), {}))
    assert "| 18034 |" in lines


def test_orchestrator_preserves_completed_training_status_if_post_train_parity_times_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    import time

    from pipeline.multi_lora_sft.config import PipelineConfig
    from pipeline.multi_lora_sft.run import Orchestrator

    cfg = PipelineConfig()
    cfg.run.output_root = str(tmp_path)
    cfg.train.sigterm_grace_sec = 0.2
    orch = Orchestrator(cfg, poll_interval_sec=0.02)
    orch.ctx.ensure_dirs()

    phase = "train:main_lora"
    final_dir = orch.ctx.checkpoints_dir / "main_lora" / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    status_path = orch.ctx.status_path(phase)

    # Child writes `state="ok"` + `final_adapter_dir` first, then simulates slow parity exceeding the hard deadline.
    snippet = (
        "import json, time; "
        f"p = {str(status_path)!r}; "
        f"d = {str(final_dir)!r}; "
        "json.dump({'phase': 'train:main_lora', 'state': 'ok', 'outputs': {'final_adapter_dir': d}, 'notes': []}, open(p, 'w')); "
        "time.sleep(10)"
    )
    monkeypatch.setattr(orch, "command_builder", lambda _p, _c, _d, _e=None: [sys.executable, "-c", snippet])
    st = orch._run_subprocess_phase(phase, time.time() + 0.35)
    assert st["state"] == "ok"
    assert st["outputs"]["final_adapter_dir"] == str(final_dir)
    assert any("Hard phase deadline reached after training completed" in n for n in st["notes"])
