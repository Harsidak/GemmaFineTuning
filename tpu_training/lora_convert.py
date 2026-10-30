"""Exact conversion between qwix/Tunix LoRA tensors and PEFT `adapter_model.safetensors` tensors (pure numpy).

Tunix applies LoRA to Gemma 4 through qwix, which stores one (A, B) pair per einsum weight. The submission needs
PEFT-style per-projection matrices (`lora_A.weight [r, in]`, `lora_B.weight [out, r]`) that vLLM can load. The two
describe the same low-rank update `delta_W = (alpha / r) * B @ A`, only laid out differently:

    qwix site                 qwix A / B shapes               PEFT module(s)
    ------------------------  ------------------------------  -----------------------------------------
    attn/q_einsum   (N,D,H)   a (D, r)        b (r, N, H)     self_attn.q_proj
    attn/kv_einsum  (2,K,D,H) a (D, r)        b (r, 2, K, H)  self_attn.k_proj + self_attn.v_proj
    attn/k_einsum   (K,D,H)   a (D, r)        b (r, K, H)     self_attn.k_proj           (K == V globals)
    attn/attn_vec_einsum (N,H,D) a (N, H, r) b (r, D)         self_attn.o_proj
    mlp/{gate,up,down}_proj   a (in, r)       b (r, out)      mlp.{gate,up,down}_proj

`kv_einsum` fuses K and V and shares one A between them; PEFT gets the same A twice and the two halves of B. Tensors
are always handled as float32 numpy arrays; casting to the serving dtype is the export phase's job.

Both directions are exact (no approximation), which `tests/test_lora_convert.py` proves by comparing the dense
`delta_W` computed from either layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

PEFT_PREFIX = "base_model.model.model.language_model.layers"
_PATH_RE = re.compile(r"(?:^|[./])layers[./](\d+)[./](attn|mlp)[./]([A-Za-z_]+)[./](w|kernel)_lora_([ab])$")

# qwix module name -> (kind)
_ATTN_KINDS = {"q_einsum": "q", "kv_einsum": "kv", "k_einsum": "k", "attn_vec_einsum": "o"}
_MLP_KINDS = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


class ConversionError(ValueError):
    """The tensors do not have the structure the converter knows how to map."""


@dataclass(frozen=True)
class Site:
    layer: int
    kind: str  # q | kv | k | o | gate | up | down

    @property
    def peft_modules(self) -> tuple[str, ...]:
        return {
            "q": ("self_attn.q_proj",), "kv": ("self_attn.k_proj", "self_attn.v_proj"), "k": ("self_attn.k_proj",),
            "o": ("self_attn.o_proj",), "gate": ("mlp.gate_proj",), "up": ("mlp.up_proj",), "down": ("mlp.down_proj",),
        }[self.kind]


def parse_qwix_path(path: str) -> tuple[Site, str] | None:
    """('layers/3/attn/q_einsum/w_lora_a') -> (Site(3,'q'), 'a'); None for anything that is not a LoRA tensor we map."""
    m = _PATH_RE.search(path.replace("\\", "/"))
    if not m:
        return None
    layer, block, mod, _w, ab = m.groups()
    kind = (_ATTN_KINDS if block == "attn" else _MLP_KINDS).get(mod)
    return (Site(int(layer), kind), ab) if kind else None


def group_sites(params: dict[str, np.ndarray]) -> dict[Site, dict[str, np.ndarray]]:
    sites: dict[Site, dict[str, np.ndarray]] = {}
    unknown: list[str] = []
    for path, arr in params.items():
        parsed = parse_qwix_path(path)
        if parsed is None:
            unknown.append(path)
            continue
        site, ab = parsed
        sites.setdefault(site, {})[ab] = np.asarray(arr, dtype=np.float32)
    if unknown:
        raise ConversionError(f"unrecognised LoRA tensor path(s), refusing to drop them silently: {unknown[:4]}")
    for site, ab in sites.items():
        if set(ab) != {"a", "b"}:
            raise ConversionError(f"{site}: expected both lora_a and lora_b, got {sorted(ab)}")
    return sites


def _peft_key(layer: int, module: str, which: str) -> str:
    return f"{PEFT_PREFIX}.{layer}.{module}.lora_{which}.weight"


def qwix_to_peft(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """qwix LoRA tensors (keyed by their path) -> PEFT tensors."""
    out: dict[str, np.ndarray] = {}
    for site, ab in sorted(group_sites(params).items(), key=lambda kv: (kv[0].layer, kv[0].kind)):
        a, b = ab["a"], ab["b"]
        k, L = site.kind, site.layer
        if k in {"q", "k"}:
            if a.ndim != 2 or b.ndim != 3 or b.shape[0] != a.shape[1]:
                raise ConversionError(f"layers.{L} {k}: unexpected shapes a{a.shape} b{b.shape}")
            r = a.shape[1]
            out[_peft_key(L, site.peft_modules[0], "A")] = np.ascontiguousarray(a.T)  # [r, D]
            out[_peft_key(L, site.peft_modules[0], "B")] = np.ascontiguousarray(b.reshape(r, -1).T)  # [heads*H, r]
        elif k == "kv":
            if a.ndim != 2 or b.ndim != 4 or b.shape[1] != 2 or b.shape[0] != a.shape[1]:
                raise ConversionError(f"layers.{L} kv: unexpected shapes a{a.shape} b{b.shape}")
            r = a.shape[1]
            for idx, module in enumerate(site.peft_modules):  # 0 -> k_proj, 1 -> v_proj
                out[_peft_key(L, module, "A")] = np.ascontiguousarray(a.T)
                out[_peft_key(L, module, "B")] = np.ascontiguousarray(b[:, idx].reshape(r, -1).T)
        elif k == "o":
            if a.ndim != 3 or b.ndim != 2 or b.shape[0] != a.shape[2]:
                raise ConversionError(f"layers.{L} o: unexpected shapes a{a.shape} b{b.shape}")
            r = a.shape[2]
            out[_peft_key(L, site.peft_modules[0], "A")] = np.ascontiguousarray(a.reshape(-1, r).T)  # [r, N*H]
            out[_peft_key(L, site.peft_modules[0], "B")] = np.ascontiguousarray(b.T)  # [D, r]
        else:  # mlp
            if a.ndim != 2 or b.ndim != 2 or b.shape[0] != a.shape[1]:
                raise ConversionError(f"layers.{L} {k}: unexpected shapes a{a.shape} b{b.shape}")
            out[_peft_key(L, site.peft_modules[0], "A")] = np.ascontiguousarray(a.T)
            out[_peft_key(L, site.peft_modules[0], "B")] = np.ascontiguousarray(b.T)
    if not out:
        raise ConversionError("no LoRA tensors to convert")
    return out


def peft_to_qwix(peft: dict[str, np.ndarray], template: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """PEFT tensors -> qwix LoRA tensors, shaped like `template` (qwix path -> shape of the live model's tensor).

    Raises `ConversionError` when the adapter cannot be represented (e.g. a PyTorch-trained adapter whose k_proj and
    v_proj have different A matrices, which the fused `kv_einsum` cannot express) or does not fit the model.
    """
    out: dict[str, np.ndarray] = {}
    sites = group_sites({p: np.zeros(s, np.float32) for p, s in template.items()})
    path_of: dict[tuple[Site, str], str] = {}
    for p in template:
        site, ab = parse_qwix_path(p)  # type: ignore[misc]
        path_of[(site, ab)] = p

    def get(layer: int, module: str, which: str) -> np.ndarray:
        key = _peft_key(layer, module, which)
        if key not in peft:
            raise ConversionError(f"adapter has no tensor {key!r}")
        return np.asarray(peft[key], dtype=np.float32)

    for site in sites:
        k, L = site.kind, site.layer
        pa, pb = path_of[(site, "a")], path_of[(site, "b")]
        sa, sb = tuple(template[pa]), tuple(template[pb])
        try:
            a, b = _site_to_qwix(site, get, sa, sb)
        except (ValueError, IndexError) as exc:
            if isinstance(exc, ConversionError):
                raise
            raise ConversionError(f"layers.{L} {k}: adapter tensors do not fit the model (a{sa}, b{sb}): {exc}") from exc
        if tuple(a.shape) != sa or tuple(b.shape) != sb:
            raise ConversionError(f"layers.{L} {k}: adapter gives a{a.shape} b{b.shape}, the model expects a{sa} b{sb}")
        out[pa], out[pb] = np.ascontiguousarray(a), np.ascontiguousarray(b)

    extra = {k for k in peft if k.endswith(".weight")} - {
        _peft_key(s.layer, m, w) for s in sites for m in s.peft_modules for w in ("A", "B")
    }
    if extra:
        raise ConversionError(f"adapter has {len(extra)} tensor(s) the model has no LoRA site for, e.g. {sorted(extra)[:3]}")
    return out


def _site_to_qwix(site: Site, get, sa: tuple[int, ...], sb: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """One site's PEFT tensors -> (qwix a, qwix b)."""
    k, L = site.kind, site.layer
    if k == "kv":
        Ak, Bk = get(L, "self_attn.k_proj", "A"), get(L, "self_attn.k_proj", "B")
        Av, Bv = get(L, "self_attn.v_proj", "A"), get(L, "self_attn.v_proj", "B")
        if not np.allclose(Ak, Av, rtol=1e-5, atol=1e-7):
            raise ConversionError(
                f"layers.{L}: k_proj and v_proj have different lora_A, but Tunix shares one A across the fused kv weight"
            )
        r = Ak.shape[0]
        return Ak.T, np.stack([Bk.T.reshape(r, *sb[2:]), Bv.T.reshape(r, *sb[2:])], axis=1)
    module = site.peft_modules[0]
    A, B = get(L, module, "A"), get(L, module, "B")
    if k in {"q", "k"}:
        return A.T, B.T.reshape(sb)
    if k == "o":
        return A.T.reshape(sa), B.T
    return A.T, B.T  # mlp


def dense_delta_from_qwix(site: Site, a: np.ndarray, b: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """Dense `delta_W` per PEFT module in HF layout (out, in), computed straight from the qwix einsum definition.

    Independent of `qwix_to_peft`; the tests compare the two to prove the conversion is exact.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    if site.kind == "q" or site.kind == "k":
        delta = np.einsum("dr,r...->...d", a, b)  # (..heads, H, D) with heads/H flattened below
        n, h = b.shape[1], b.shape[2]
        return {site.peft_modules[0]: scale * delta.reshape(n * h, a.shape[0])}
    if site.kind == "kv":
        res = {}
        for idx, module in enumerate(site.peft_modules):
            bi = b[:, idx]  # (r, K, H)
            delta = np.einsum("dr,rkh->khd", a, bi)
            res[module] = scale * delta.reshape(bi.shape[1] * bi.shape[2], a.shape[0])
        return res
    if site.kind == "o":
        delta = np.einsum("nhr,rd->dnh", a, b)  # HF o_proj.weight is (D, N*H)
        return {site.peft_modules[0]: scale * delta.reshape(b.shape[1], -1)}
    return {site.peft_modules[0]: scale * (a @ b).T}  # mlp: qwix kernel is (in, out); HF weight is (out, in)
