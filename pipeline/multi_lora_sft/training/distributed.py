"""Multi-process helpers for `train.parallelism = "fsdp"` (FSDP1 + QLoRA) and DDP.

FSDP shards the frozen 4-bit base, the LoRA weights, their gradients and the Adam moments across all GPUs, so every
GPU computes its own micro-batch at the same time (unlike `model_split`, where the GPUs take turns). That makes a few
things collective that are single-process elsewhere in the pipeline:

- Reading weights: a LoRA weight is split over the ranks, so saving an adapter (checkpoint, best, final) or checking
  for dead LoRA modules first gathers the LoRA tensors on rank 0. `gather_lora_state_dict` all-gathers only the LoRA
  FSDP units (the PEFT wrap policy gives every LoRA linear its own unit), never the ~18 GB base.
- Optimizer state: `fsdp_full_optim_state_dict` / `fsdp_optim_state_to_load` convert between the per-rank flat-param
  shards and one name-keyed state dict, so a checkpoint resumes on any world size.
- Decisions: every rank must take the same branch around a collective, or the run hangs. `SyncedClock` gives all
  ranks rank 0's wall clock (WSD schedule, checkpoint/validation timers, deadline stop) and `agree_any` /
  `agree_all` combine per-rank flags (SIGTERM, time left).

Everything is a no-op when torch.distributed is not initialised, so single-process paths are unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

GEMMA4_DECODER_LAYER_CLS = "Gemma4TextDecoderLayer"

# Accelerate builds its FSDP plugin from these when no plugin is passed in (TrainingArguments.fsdp stays unset, so the
# same code path works on every transformers 5.x). FSDP1: bitsandbytes 4-bit weights are flattened into FSDP's flat
# parameters, which FSDP2's per-parameter DTensor sharding does not support.
FSDP_ENV: dict[str, str] = {
    "ACCELERATE_USE_FSDP": "true",
    "FSDP_VERSION": "1",
    "FSDP_SHARDING_STRATEGY": "FULL_SHARD",
    # The real policy (decoder layers + one unit per LoRA linear) is installed by the Trainer via PEFT's
    # `fsdp_auto_wrap_policy`, which reads FSDP_TRANSFORMER_CLS_TO_WRAP.
    "FSDP_AUTO_WRAP_POLICY": "TRANSFORMER_BASED_WRAP",
    "FSDP_TRANSFORMER_CLS_TO_WRAP": GEMMA4_DECODER_LAYER_CLS,
    "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
    "FSDP_FORWARD_PREFETCH": "false",
    # Flat params must have one requires_grad value; the LoRA units are separate, so the original-params mode is unused.
    "FSDP_USE_ORIG_PARAMS": "false",
    # Every rank loads the same checkpoint and seeds LoRA identically; broadcasting ~18 GB from rank 0 buys nothing.
    "FSDP_SYNC_MODULE_STATES": "false",
    "FSDP_CPU_RAM_EFFICIENT_LOADING": "false",
    "FSDP_OFFLOAD_PARAMS": "false",
    "FSDP_STATE_DICT_TYPE": "FULL_STATE_DICT",
    # HF gradient checkpointing (TrainingArguments.gradient_checkpointing) is used instead; both cannot be on.
    "FSDP_ACTIVATION_CHECKPOINTING": "false",
}


def apply_fsdp_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Sets the accelerate FSDP plugin environment (overwriting any launcher value) and returns what was set."""
    env = os.environ if environ is None else environ
    for key, val in FSDP_ENV.items():
        env[key] = val
    return dict(FSDP_ENV)


def dist_ready() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    if dist_ready():
        import torch.distributed as dist

        return int(dist.get_world_size())
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def rank() -> int:
    if dist_ready():
        import torch.distributed as dist

        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def init_distributed() -> None:
    """Initialises the process group (NCCL on GPUs) and binds this process to its local GPU, once.

    Done before model loading so all ranks can agree on the phase start time and each rank loads onto its own GPU.
    The Trainer's Accelerator later reuses this process group."""
    if dist_ready() or world_size() <= 1:
        return
    from accelerate import PartialState

    PartialState()


def _collective_device() -> Any:
    import torch
    import torch.distributed as dist

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_float(value: float, src: int = 0) -> float:
    """Rank `src`'s value on every rank (float64, so epoch timestamps keep sub-millisecond precision)."""
    if not dist_ready() or world_size() <= 1:
        return float(value)
    import torch
    import torch.distributed as dist

    t = torch.tensor([float(value)], dtype=torch.float64, device=_collective_device())
    dist.broadcast(t, src=src)
    return float(t.item())


def broadcast_int(value: int, src: int = 0) -> int:
    if not dist_ready() or world_size() <= 1:
        return int(value)
    import torch
    import torch.distributed as dist

    t = torch.tensor([int(value)], dtype=torch.int64, device=_collective_device())
    dist.broadcast(t, src=src)
    return int(t.item())


def agree_any(flag: bool) -> bool:
    """True on every rank when it is True on at least one rank."""
    if not dist_ready() or world_size() <= 1:
        return bool(flag)
    import torch
    import torch.distributed as dist

    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=_collective_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def agree_all(flag: bool) -> bool:
    """True on every rank only when it is True on all ranks."""
    return not agree_any(not flag)


def barrier() -> None:
    if dist_ready() and world_size() > 1:
        import torch.distributed as dist

        dist.barrier()


class SyncedClock:
    """A wall clock that reads the same on every rank.

    Single process: plain `time.time()`. Multi-process: the value rank 0 broadcast at the last `sync()` (called once per
    optimizer step by the training callback). Time-based decisions taken between syncs (the WSD learning-rate lambda)
    therefore see a timestamp that is at most one step old, but identical across ranks."""

    def __init__(self, now_fn: Callable[[], float] = time.time) -> None:
        self._now_fn = now_fn
        self._value: float | None = None

    @property
    def distributed(self) -> bool:
        return dist_ready() and world_size() > 1

    def sync(self) -> float:
        """Collective in multi-process runs: every rank must call it at the same point."""
        if not self.distributed:
            return float(self._now_fn())
        self._value = broadcast_float(self._now_fn())
        return self._value

    def __call__(self) -> float:
        if not self.distributed:
            return float(self._now_fn())
        if self._value is None:
            return self.sync()
        return self._value


# --------------------------------------------------------------------------------------------- FSDP model access


def _fsdp_cls() -> Any:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:  # torch built without distributed
        return None
    return FullyShardedDataParallel


def is_fsdp_wrapped(model: Any) -> bool:
    fsdp = _fsdp_cls()
    return fsdp is not None and model is not None and isinstance(model, fsdp)


def _clean_name(name: str) -> str:
    for token in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
        name = name.replace(token, "")
    return name


def _join(*parts: str) -> str:
    return ".".join(p for p in parts if p)


def _trainable_units(model: Any) -> list[tuple[str, Any]]:
    """(name relative to the wrapped PeftModel, FSDP unit) for every FSDP unit holding a trainable flat parameter."""
    fsdp = _fsdp_cls()
    units: list[tuple[str, Any]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, fsdp):
            continue
        handle = getattr(mod, "_handle", None)
        flat = getattr(handle, "flat_param", None) if handle is not None else None
        if flat is None or not flat.requires_grad:
            continue
        units.append((_clean_name(name), mod))
    return units


def gather_lora_state_dict(model: Any, *, key_filter: str = "lora_") -> dict[str, Any] | None:
    """Full-precision CPU copy of the trainable LoRA tensors, keyed like `PeftModel.state_dict()`.

    Collective for an FSDP-wrapped model: every rank must call it at the same point; rank 0 receives the dict and the
    other ranks get `{}`. Returns None for a model that is not FSDP-wrapped (callers then read the module directly)."""
    if not is_fsdp_wrapped(model):
        return None
    fsdp = _fsdp_cls()
    is_rank0 = rank() == 0
    out: dict[str, Any] = {}
    for prefix, unit in _trainable_units(model):
        flat = unit._handle.flat_param
        with fsdp.summon_full_params(unit, recurse=False, writeback=False, rank0_only=True, offload_to_cpu=True):
            if not is_rank0:
                continue
            for info in flat._param_infos:
                key = _join(prefix, info.module_name, info.param_name)
                if key_filter and key_filter not in key:
                    continue
                out[key] = getattr(info.module, info.param_name).detach().to("cpu").clone()
    return out


def fsdp_full_optim_state_dict(model: Any, optimizer: Any) -> dict[str, Any] | None:
    """Name-keyed (world-size independent) optimizer state on rank 0, `{}` elsewhere. Collective."""
    if not is_fsdp_wrapped(model) or optimizer is None:
        return None
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )

    fsdp = _fsdp_cls()
    opt = getattr(optimizer, "optimizer", optimizer)  # accelerate's AcceleratedOptimizer wraps the torch one
    with fsdp.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        osd = fsdp.optim_state_dict(model, opt)
    return osd if rank() == 0 else {}


def fsdp_optim_state_to_load(model: Any, optimizer: Any, full_osd: dict[str, Any]) -> dict[str, Any]:
    """Converts a name-keyed full optimizer state (every rank passes the same dict) to this rank's shard. Collective."""
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )

    fsdp = _fsdp_cls()
    opt = getattr(optimizer, "optimizer", optimizer)
    with fsdp.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        return fsdp.optim_state_dict_to_load(model, opt, full_osd)


def keep_buffers_in_native_dtype(fsdp_plugin: Any) -> None:
    """Stops FSDP mixed precision from casting buffers (FSDP1 casts them to bf16 at the first forward by default).

    Gemma 4 keeps its RoPE `inv_freq` in fp32; rounding it to bf16 shifts the rotary phase by tens of radians at 16k
    positions, so the adapter would train against different positions than vLLM serves. Wraps the plugin's
    `set_mixed_precision` (the Trainer calls it again for QLoRA right before wrapping) so every policy it builds has
    `buffer_dtype=None`, and fixes the current one."""
    if fsdp_plugin is None or getattr(fsdp_plugin, "_buffers_native_patch", False):
        return

    def _strip() -> None:
        pol = getattr(fsdp_plugin, "mixed_precision_policy", None)
        if pol is not None and getattr(pol, "buffer_dtype", None) is not None and dataclasses.is_dataclass(pol):
            fsdp_plugin.mixed_precision_policy = dataclasses.replace(pol, buffer_dtype=None)

    original = fsdp_plugin.set_mixed_precision

    def _patched(*args: Any, **kwargs: Any) -> Any:
        res = original(*args, **kwargs)
        _strip()
        return res

    fsdp_plugin.set_mixed_precision = _patched
    fsdp_plugin._buffers_native_patch = True
    _strip()
