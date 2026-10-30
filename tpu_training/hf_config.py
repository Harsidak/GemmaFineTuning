"""Hugging Face Gemma 4 `config.json` -> Tunix `ModelConfig`.

Tunix ships verified presets for E2B / E4B / 31B. When the checkpoint's dimensions equal a preset we use the preset
(Google's tested values) and only overlay training-time settings; otherwise (tiny test fixtures, other sizes) the
config is derived field by field from the checkpoint. Either way every dimension that must agree with the weights is
checked against `config.json`, so a mismatch fails here with a readable message instead of as a shape error deep in
the loader.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRESET_KEYS = ("num_layers", "embed_dim", "hidden_dim", "num_heads", "head_dim", "num_kv_heads")


class ModelConfigError(ValueError):
    pass


def load_text_config(model_dir: str | Path) -> dict[str, Any]:
    """The `text_config` section (or the whole file for a text-only checkpoint)."""
    p = Path(model_dir) / "config.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelConfigError(f"cannot read {p}: {exc}") from exc
    cfg = raw.get("text_config", raw)
    if not isinstance(cfg, dict) or "hidden_size" not in cfg:
        raise ModelConfigError(f"{p} has no Gemma text config (no 'hidden_size')")
    return cfg


def _pattern(layer_types: list[str], attention_type: Any) -> tuple[Any, ...]:
    mapping = {"sliding_attention": attention_type.LOCAL_SLIDING, "full_attention": attention_type.GLOBAL}
    try:
        seq = [mapping[t] for t in layer_types]
    except KeyError as exc:
        raise ModelConfigError(f"unknown layer type {exc} in config.json layer_types") from exc
    for period in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return tuple(seq[:period])
    return tuple(seq)  # pragma: no cover - the full list is always a valid period


def derive_fields(hf: dict[str, Any]) -> dict[str, Any]:
    """Everything `config.json` states, in Tunix field names."""
    from tunix.models.gemma4 import model as M

    if hf.get("enable_moe_block") or hf.get("num_experts"):
        raise ModelConfigError("MoE Gemma 4 variants are not supported by this trainer (E4B / E2B / 31B are dense)")
    n = int(hf["num_hidden_layers"])
    layer_types = hf.get("layer_types") or []
    if len(layer_types) != n:
        raise ModelConfigError(f"layer_types has {len(layer_types)} entries for {n} layers")
    pattern = _pattern(layer_types, M.AttentionType)

    glob_dim = hf.get("global_head_dim")
    if glob_dim is None:
        per_layer = hf.get("per_layer_config") or {}
        dims = {int(v["head_dim"]) for i, v in per_layer.items() if "head_dim" in v and layer_types[int(i)] == "full_attention"}
        glob_dim = dims.pop() if len(dims) == 1 else hf.get("head_dim")
    rope = hf.get("rope_parameters") or {}
    local_rope, global_rope = rope.get("sliding_attention", {}), rope.get("full_attention", {})
    out: dict[str, Any] = {
        "num_layers": n,
        "num_embed": int(hf["vocab_size"]),
        "embed_dim": int(hf["hidden_size"]),
        "hidden_dim": int(hf["intermediate_size"]),
        "num_heads": int(hf["num_attention_heads"]),
        "head_dim": int(hf["head_dim"]),
        "num_kv_heads": int(hf["num_key_value_heads"]),
        "final_logit_softcap": hf.get("final_logit_softcapping"),
        "sliding_window_size": hf.get("sliding_window"),
        "per_layer_input_dim": int(hf.get("hidden_size_per_layer_input") or 0),
        "frac_shared_layers": float(hf.get("num_kv_shared_layers") or 0) / n,
        "global_key_size": int(glob_dim) if glob_dim else None,
        "num_global_kv_heads": hf.get("num_global_key_value_heads"),
        "attention_pattern": pattern,
        "k_eq_v_global": bool(hf.get("attention_k_eq_v", False)),
    }
    if "rope_theta" in local_rope:
        out["local_base_frequency"] = int(local_rope["rope_theta"])
    if "rope_theta" in global_rope:
        out["global_base_frequency"] = int(global_rope["rope_theta"])
    if "partial_rotary_factor" in global_rope:
        out["global_rope_proportion"] = float(global_rope["partial_rotary_factor"])
    if hf.get("use_double_wide_mlp"):
        out["override_kv_shared_ffw_hidden"] = 2 * int(hf["intermediate_size"])
    return out


def build_model_config(
    hf: dict[str, Any],
    *,
    param_dtype: Any,
    dtype: Any,
    remat: str = "BLOCK",
    flash_attention: bool = False,
) -> Any:
    """Tunix `ModelConfig` for the checkpoint, set up for training (remat on, no KV cache)."""
    from tunix.models.gemma4 import model as M

    derived = derive_fields(hf)
    config_cls = M.ModelConfig
    presets = {"e4b": config_cls.gemma4_e4b, "e2b": config_cls.gemma4_e2b, "31b": config_cls.gemma4_31b}
    base = None
    for name, ctor in presets.items():
        preset = ctor()
        if all(getattr(preset, k) == derived[k] for k in PRESET_KEYS) and preset.num_embed == derived["num_embed"]:
            base = preset
            logger.info("Checkpoint matches the Tunix %s preset", name)
            diffs = {
                k: (getattr(preset, k), v) for k, v in derived.items()
                if k != "attention_pattern" and v is not None and getattr(preset, k) != v
            }
            if diffs:
                logger.warning("config.json differs from the Tunix %s preset in %s; keeping the preset", name, diffs)
            break
    train_overrides = dict(
        param_dtype=param_dtype,
        dtype=dtype,
        remat_config=M.RematConfig[remat],
        use_flash_attention=bool(flash_attention),
        use_sliding_window_kv_cache=False,  # training never uses the decode cache; the sliding-window cache excludes flash
    )
    if base is not None:
        return dataclasses.replace(base, **train_overrides)
    fields = {k: v for k, v in derived.items() if v is not None}
    try:
        return config_cls(**fields, **train_overrides)
    except TypeError as exc:
        raise ModelConfigError(f"could not build a Tunix ModelConfig from config.json: {exc}") from exc


def check_layout_against_checkpoint(model: Any, model_dir: str | Path) -> None:
    """The layers Tunix treats as KV-shared must be exactly those without k_proj in the checkpoint (E4B/E2B)."""
    import re

    idx_file = Path(model_dir) / "model.safetensors.index.json"
    if idx_file.is_file():
        names = list(json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {}))
    else:
        from pipeline.multi_lora_sft.shared.checkpoints import read_safetensors_header

        names = [n for f in sorted(Path(model_dir).glob("*.safetensors")) for n in read_safetensors_header(f)["tensors"]]
    owners = {int(m.group(1)) for n in names if "vision" not in n and "audio" not in n
              for m in [re.search(r"layers\.(\d+)\.self_attn\.k_proj\.weight$", n)] if m}
    shared = {i for i, j in enumerate(model.kv_cache_sharing_patterns) if i != j}
    want_shared = set(range(model.config.num_layers)) - owners
    if shared != want_shared:
        raise ModelConfigError(
            f"KV-sharing layout mismatch: Tunix shares layers {sorted(shared)}, the checkpoint has no k_proj on {sorted(want_shared)}"
        )
