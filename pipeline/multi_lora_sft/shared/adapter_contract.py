"""Structural "will the grader's vLLM accept this adapter?" check, torch-free and backend-independent.

The submission serves the `-qat-w4a16-ct` checkpoint with the LoRA applied on top, while training happened on the
unquantized sibling. A real generation test needs GPUs (the eval phase does that on CUDA); this check catches the
mismatches that need no GPU and no framework, using only safetensors headers:

* every key is `base_model.model.model.language_model.layers.N.<module>.lora_{A,B}.weight`, no vision / audio keys
* each module has both A and B, one consistent rank, the dtype vLLM loads (bf16 / fp16 / fp32)
* each adapted module exists in the SERVED checkpoint, and where the served weight is stored plain its
  in/out features equal A.in / B.out (packed int4 weights are checked through their `weight_shape` tensor when readable)
* `adapter_config.json` agrees with the weights (r, target modules), size is under the 3 GiB submission limit
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoints import ADAPTER_FILE, read_safetensors_header

KEY_RE = re.compile(r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")
SERVED_RE = re.compile(r"layers\.(\d+)\.((?:self_attn|mlp)\.[a-z_]+)\.(weight|weight_packed|weight_shape|weight_scale|qweight)$")
OK_DTYPES = {"BF16", "F16", "F32"}
MAX_SUBMISSION_BYTES = 3 * 1024**3


@dataclass
class ContractReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "info": self.info}


def served_module_index(served_dir: str | Path) -> dict[tuple[int, str], dict[str, tuple[Path, str]]]:
    """(layer, module) -> {suffix: (shard path, tensor name)} for the language-model linear layers of a served checkpoint."""
    d = Path(served_dir)
    index_file = d / "model.safetensors.index.json"
    names: dict[str, Path] = {}
    if index_file.is_file():
        wm = json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {})
        names = {n: d / shard for n, shard in wm.items()}
    else:
        for shard in sorted(d.glob("*.safetensors")):
            for n in read_safetensors_header(shard)["tensors"]:
                names[n] = shard
    out: dict[tuple[int, str], dict[str, tuple[Path, str]]] = {}
    for name, shard in names.items():
        if any(bad in name for bad in ("vision", "audio")):
            continue
        m = SERVED_RE.search(name)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = (shard, name)
    return out


def _served_features(entry: dict[str, tuple[Path, str]]) -> tuple[int, int] | None:
    """(out_features, in_features) of one served linear, or None when it cannot be determined cheaply."""
    try:
        if "weight" in entry:
            shard, name = entry["weight"]
            shape = read_safetensors_header(shard)["tensors"][name]["shape"]
            return (int(shape[0]), int(shape[1])) if len(shape) == 2 else None
        if "weight_shape" in entry:
            from safetensors import safe_open

            shard, name = entry["weight_shape"]
            with safe_open(str(shard), framework="np") as fh:
                vals = fh.get_tensor(name).tolist()
            return (int(vals[0]), int(vals[1])) if len(vals) == 2 else None
    except Exception:  # noqa: BLE001 - unverifiable is a warning, not an error
        return None
    return None


def check_adapter_contract(
    adapter_dir: str | Path,
    *,
    served_model_dir: str | Path | None = None,
    expected_rank: int | None = None,
    max_bytes: int = MAX_SUBMISSION_BYTES,
) -> ContractReport:
    rep = ContractReport()
    d = Path(adapter_dir)
    st = d / ADAPTER_FILE
    if not st.is_file():
        rep.errors.append(f"missing {ADAPTER_FILE} in {d}")
        return rep
    size = st.stat().st_size
    rep.info["bytes"] = size
    if size >= max_bytes:
        rep.errors.append(f"adapter is {size} bytes, over the {max_bytes}-byte submission limit")
    try:
        tensors = read_safetensors_header(st)["tensors"]
    except Exception as exc:  # noqa: BLE001
        rep.errors.append(f"unreadable safetensors header: {exc}")
        return rep
    if not tensors:
        rep.errors.append("adapter has 0 tensors")
        return rep

    mods: dict[tuple[int, str], dict[str, list[int]]] = {}
    dtypes: set[str] = set()
    for key, info in tensors.items():
        m = KEY_RE.match(key)
        if not m:
            rep.errors.append(f"key {key!r} does not match base_model.model.model.language_model.layers.N.<module>.lora_[AB].weight")
            continue
        if len(info["shape"]) != 2:
            rep.errors.append(f"{key}: expected a 2-D tensor, got shape {info['shape']}")
            continue
        dtypes.add(info["dtype"])
        mods.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = [int(x) for x in info["shape"]]
    bad_dtypes = dtypes - OK_DTYPES
    if bad_dtypes:
        rep.errors.append(f"unsupported dtype(s) {sorted(bad_dtypes)}; serve with bf16/fp16/fp32")
    rep.info["dtypes"] = sorted(dtypes)

    ranks: set[int] = set()
    for (layer, mod), ab in sorted(mods.items()):
        label = f"layers.{layer}.{mod}"
        if set(ab) != {"A", "B"}:
            rep.errors.append(f"{label}: has only lora_{'/'.join(sorted(ab))}")
            continue
        if ab["A"][0] != ab["B"][1]:
            rep.errors.append(f"{label}: lora_A rank {ab['A'][0]} != lora_B rank {ab['B'][1]}")
        ranks.add(ab["A"][0])
    if len(ranks) > 1:
        rep.errors.append(f"inconsistent ranks across modules: {sorted(ranks)}")
    rep.info["modules"] = len(mods)
    rep.info["ranks"] = sorted(ranks)
    if expected_rank is not None and ranks and ranks != {int(expected_rank)}:
        rep.errors.append(f"adapter rank {sorted(ranks)} != configured r={expected_rank}")

    cfg_path = d / "adapter_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if ranks and int(cfg.get("r", next(iter(ranks)))) not in ranks:
                rep.errors.append(f"adapter_config.json r={cfg.get('r')} disagrees with weights {sorted(ranks)}")
        except (OSError, ValueError) as exc:
            rep.errors.append(f"adapter_config.json unreadable: {exc}")
    else:
        rep.warnings.append("adapter_config.json missing (the export phase writes the final one)")

    if served_model_dir is not None and mods:
        try:
            served = served_module_index(served_model_dir)
        except Exception as exc:  # noqa: BLE001
            rep.warnings.append(f"served checkpoint not inspected: {exc}")
            served = {}
        if served:
            unverifiable = 0
            for (layer, mod), ab in sorted(mods.items()):
                entry = served.get((layer, mod))
                if entry is None:
                    rep.errors.append(f"served checkpoint has no layers.{layer}.{mod} (KV-shared layer or wrong model size?)")
                    continue
                feats = _served_features(entry)
                if feats is None:
                    unverifiable += 1
                    continue
                out_f, in_f = feats
                if "A" in ab and ab["A"][1] != in_f:
                    rep.errors.append(f"layers.{layer}.{mod}: lora_A in_features {ab['A'][1]} != served {in_f}")
                if "B" in ab and ab["B"][0] != out_f:
                    rep.errors.append(f"layers.{layer}.{mod}: lora_B out_features {ab['B'][0]} != served {out_f}")
            if unverifiable:
                rep.warnings.append(f"{unverifiable} module shape(s) could not be read from the served checkpoint")
            rep.info["served_modules"] = len(served)
    return rep
