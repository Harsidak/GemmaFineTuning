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

        # ---------------------------------------------------------------- lockstep greedy parity decode
        # (AdamW stepped the FSDP copy once more; give the reference the same weights before comparing)
        gathered = gather_lora_state_dict(model)
        if rank == 0:
            ref.load_state_dict(gathered, strict=False)
        prompt = batch["input_ids"][0, :10].tolist()
        tok = SimpleNamespace(eos_token_id=None, decode=lambda ids, skip_special_tokens=False: str(ids))
        with_adapter = greedy_decode_with_logprobs(model, tok, prompt, gen_tokens=5)
        with model.module.disable_adapter():
            base = greedy_decode_with_logprobs(model, tok, prompt, gen_tokens=5)
        res["parity_tokens"] = [s["token_id"] for s in with_adapter]
        res["parity_base_tokens"] = [s["token_id"] for s in base]
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
