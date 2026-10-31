"""The qwix <-> PEFT conversion must be exact: same dense weight update from either layout. Pure numpy."""

from __future__ import annotations

import numpy as np
import pytest

from tpu_training.lora_convert import (
    ConversionError,
    Site,
    dense_delta_from_qwix,
    group_sites,
    parse_qwix_path,
    peft_to_qwix,
    qwix_to_peft,
)

D, R, N, H, K, F = 24, 4, 3, 8, 2, 40
SCALE = 2.0  # alpha / r


def _model_params(rng: np.random.Generator, layers: int = 3) -> dict[str, np.ndarray]:
    """Random LoRA tensors with exactly the shapes qwix produces for a tiny Gemma 4 (see the probe in the docstring)."""
    t: dict[str, np.ndarray] = {}
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    for L in range(layers):
        p = f"layers/{L}"
        t[f"{p}/attn/q_einsum/w_lora_a"], t[f"{p}/attn/q_einsum/w_lora_b"] = r(D, R), r(R, N, H)
        t[f"{p}/attn/attn_vec_einsum/w_lora_a"], t[f"{p}/attn/attn_vec_einsum/w_lora_b"] = r(N, H, R), r(R, D)
        if L < layers - 1:  # the last layer reuses another layer's KV: no k/v adapters
            t[f"{p}/attn/kv_einsum/w_lora_a"], t[f"{p}/attn/kv_einsum/w_lora_b"] = r(D, R), r(R, 2, K, H)
        t[f"{p}/mlp/gate_proj/kernel_lora_a"], t[f"{p}/mlp/gate_proj/kernel_lora_b"] = r(D, R), r(R, F)
        t[f"{p}/mlp/up_proj/kernel_lora_a"], t[f"{p}/mlp/up_proj/kernel_lora_b"] = r(D, R), r(R, F)
        t[f"{p}/mlp/down_proj/kernel_lora_a"], t[f"{p}/mlp/down_proj/kernel_lora_b"] = r(F, R), r(R, D)
    return t


def test_parse_paths() -> None:
    assert parse_qwix_path("layers/3/attn/q_einsum/w_lora_a") == (Site(3, "q"), "a")
    assert parse_qwix_path("model.layers.12.mlp.down_proj.kernel_lora_b") == (Site(12, "down"), "b")
    assert parse_qwix_path("layers/0/attn/kv_einsum/w_lora_b")[0].peft_modules == ("self_attn.k_proj", "self_attn.v_proj")
    assert parse_qwix_path("embedder/per_layer_projection/w_lora_a") is None


def test_peft_layout_matches_hf_shapes() -> None:
    peft = qwix_to_peft(_model_params(np.random.default_rng(0)))
    pre = "base_model.model.model.language_model.layers.0"
    assert peft[f"{pre}.self_attn.q_proj.lora_A.weight"].shape == (R, D)
    assert peft[f"{pre}.self_attn.q_proj.lora_B.weight"].shape == (N * H, R)
    assert peft[f"{pre}.self_attn.k_proj.lora_B.weight"].shape == (K * H, R)
    assert peft[f"{pre}.self_attn.v_proj.lora_A.weight"].shape == (R, D)
    assert peft[f"{pre}.self_attn.o_proj.lora_A.weight"].shape == (R, N * H)
    assert peft[f"{pre}.self_attn.o_proj.lora_B.weight"].shape == (D, R)
    assert peft[f"{pre}.mlp.down_proj.lora_A.weight"].shape == (R, F)
    assert peft[f"{pre}.mlp.down_proj.lora_B.weight"].shape == (D, R)
    assert not any(k.startswith("base_model.model.model.language_model.layers.2.self_attn.k_proj") for k in peft)
    assert all(v.dtype == np.float32 and v.flags["C_CONTIGUOUS"] for v in peft.values())


def test_dense_update_is_identical_in_both_layouts() -> None:
    params = _model_params(np.random.default_rng(1))
    peft = qwix_to_peft(params)
    checked = 0
    for site, ab in group_sites(params).items():
        want = dense_delta_from_qwix(site, ab["a"], ab["b"], SCALE)
        for module, delta in want.items():
            key = f"base_model.model.model.language_model.layers.{site.layer}.{module}"
            got = SCALE * (peft[f"{key}.lora_B.weight"].astype(np.float64) @ peft[f"{key}.lora_A.weight"].astype(np.float64))
            np.testing.assert_allclose(got, delta, rtol=1e-6, atol=1e-8, err_msg=f"{site} {module}")
            checked += 1
    assert checked == 19  # 3 layers x (q, o, gate, up, down) + k and v on the 2 layers that own a KV cache


def test_roundtrip_is_lossless() -> None:
    params = _model_params(np.random.default_rng(2))
    template = {k: v.shape for k, v in params.items()}
    back = peft_to_qwix(qwix_to_peft(params), template)
    assert back.keys() == params.keys()
    for k in params:
        np.testing.assert_array_equal(back[k], params[k], err_msg=k)


def test_unknown_or_unpaired_tensors_are_refused_not_dropped() -> None:
    params = _model_params(np.random.default_rng(3), layers=1)
    with pytest.raises(ConversionError, match="unrecognised"):
        qwix_to_peft({**params, "embedder/foo/w_lora_a": np.zeros((2, 2), np.float32)})
    broken = dict(params)
    broken.pop("layers/0/attn/q_einsum/w_lora_b")
    with pytest.raises(ConversionError, match="both lora_a and lora_b"):
        qwix_to_peft(broken)
    bad_shape = dict(params)
    bad_shape["layers/0/attn/q_einsum/w_lora_b"] = np.zeros((R + 1, N, H), np.float32)
    with pytest.raises(ConversionError, match="unexpected shapes"):
        qwix_to_peft(bad_shape)
    with pytest.raises(ConversionError):
        qwix_to_peft({})


def test_pytorch_trained_adapter_with_distinct_kv_A_cannot_be_loaded_into_the_fused_kv_weight() -> None:
    params = _model_params(np.random.default_rng(4), layers=2)
    template = {k: v.shape for k, v in params.items()}
    peft = qwix_to_peft(params)
    pre = "base_model.model.model.language_model.layers.0.self_attn"
    peft[f"{pre}.v_proj.lora_A.weight"] = peft[f"{pre}.v_proj.lora_A.weight"] + 1.0  # what an independent v_proj A looks like
    with pytest.raises(ConversionError, match="different lora_A"):
        peft_to_qwix(peft, template)


def test_peft_to_qwix_checks_fit_and_completeness() -> None:
    params = _model_params(np.random.default_rng(5), layers=2)
    template = {k: v.shape for k, v in params.items()}
    peft = qwix_to_peft(params)
    missing = dict(peft)
    missing.pop("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight")
    with pytest.raises(ConversionError, match="no tensor"):
        peft_to_qwix(missing, template)
    wrong_rank_template = {k: (s if "q_einsum" not in k else (s[0] + 1, *s[1:])) for k, s in template.items()}
    with pytest.raises(ConversionError):
        peft_to_qwix(peft, wrong_rank_template)
    extra = dict(peft)
    extra["base_model.model.model.language_model.layers.9.mlp.up_proj.lora_A.weight"] = np.zeros((R, D), np.float32)
    with pytest.raises(ConversionError, match="no LoRA site"):
        peft_to_qwix(extra, template)
