"""Unit tests for `pipeline.multi_lora_sft.training.loss` (design_doc.md §7.3, §7.2, §11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Gemma4ForConditionalGeneration

from pipeline.multi_lora_sft.training.data import compute_token_weights
from pipeline.multi_lora_sft.training.loss import (
    WeightedLanguageModelingCollator,
    get_model_softcap,
    selective_cross_entropy_loss,
)
from pipeline.src.data.processor import IGNORE_INDEX, tokenize_trajectory


def test_selective_loss_matches_gemma4_builtin_loss(tiny_gemma4_dir: Path) -> None:
    model = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.float32)
    model.eval()
    softcap = get_model_softcap(model)
    assert softcap == pytest.approx(30.0)

    torch.manual_seed(7)
    input_ids = torch.randint(4, 200, (2, 24), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :8] = IGNORE_INDEX
    labels[1, 18:] = IGNORE_INDEX

    out = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
    builtin_loss = out.loss

    # Force small chunk_rows=5 to exercise multi-chunk accumulation + softcap
    last_hidden = out.hidden_states[-1]
    sel_loss = selective_cross_entropy_loss(
        last_hidden,
        model.lm_head,
        labels,
        softcap=softcap,
        chunk_rows=5,
    )
    assert torch.isfinite(sel_loss)
    assert abs(float(sel_loss.item()) - float(builtin_loss.item())) < 1e-4


def test_tool_call_token_weighting_targets_exact_span(tiny_tokenizer: Any) -> None:
    messages = [
        {"role": "user", "content": "Inspect config.py"},
        {
            "role": "assistant",
            "reasoning": "Read the file first.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "config.py"}},
                }
            ],
        },
        {"role": "tool", "name": "read_file", "content": '{"status": "ok"}'},
        {"role": "assistant", "reasoning": "Done.", "content": "All good."},
    ]
    rec = tokenize_trajectory(
        messages,
        tiny_tokenizer,
        chat_template=tiny_tokenizer.chat_template,
        max_length=2048,
    )
    assert rec is not None
    w = compute_token_weights(rec["input_ids"], rec["labels"], tiny_tokenizer, tool_call_weight=1.75)

    start_id = tiny_tokenizer.convert_tokens_to_ids("<|tool_call>")
    end_id = tiny_tokenizer.convert_tokens_to_ids("<tool_call|>")
    assert start_id in rec["input_ids"]
    assert end_id in rec["input_ids"]

    s_idx = rec["input_ids"].index(start_id)
    e_idx = rec["input_ids"].index(end_id)
    assert all(val == pytest.approx(1.75) for val in w[s_idx : e_idx + 1])
    # Final plain-text assistant turn has weight 1.0, and user/tool prompt tokens have weight 0.0
    assert any(val == pytest.approx(1.0) for val in w[e_idx + 1 :])
    assert all(
        (val == 0.0) if lbl == IGNORE_INDEX else (val in {1.0, 1.75})
        for val, lbl in zip(w, rec["labels"], strict=True)
    )


def test_grad_accum_step_normalization_invariance(tiny_gemma4_dir: Path) -> None:
    """Two unequal-length micro-batches normalized by global `weight_sum` match single combined batch gradients."""
    model_comb = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.float32)
    model_ga = Gemma4ForConditionalGeneration.from_pretrained(str(tiny_gemma4_dir), dtype=torch.float32)
    model_ga.load_state_dict(model_comb.state_dict())
    softcap = get_model_softcap(model_comb)

    ex_short = {
        "input_ids": [2, 10, 11, 12, 13, 1],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, 11, 12, 13, 1],  # 4 supervised targets
        "weights": [0.0, 0.0, 1.0, 1.5, 1.5, 1.0],
    }
    ex_long = {
        "input_ids": [2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1],  # 11 supervised targets
        "weights": [0.0, 0.0, *([1.0] * 11)],
    }

    collator = WeightedLanguageModelingCollator(pad_token_id=0)
    b1 = collator([ex_short])
    b2 = collator([ex_long])
    b_all = collator([ex_short, ex_long])

    # Total weight sum across the entire optimizer step (shifted positions 1..)
    total_w = float(
        (b1["weights"][:, 1:] * b1["labels"][:, 1:].ne(IGNORE_INDEX)).sum().item()
        + (b2["weights"][:, 1:] * b2["labels"][:, 1:].ne(IGNORE_INDEX)).sum().item()
    )

    # 1. Combined batch backward
    h_all = model_comb.model(input_ids=b_all["input_ids"], attention_mask=b_all["attention_mask"]).last_hidden_state
    loss_all = selective_cross_entropy_loss(
        h_all, model_comb.lm_head, b_all["labels"], weights=b_all["weights"], softcap=softcap, weight_sum=total_w
    )
    loss_all.backward()

    # 2. Gradient accumulation over 2 unequal micro-batches using global `weight_sum`
    h1 = model_ga.model(input_ids=b1["input_ids"], attention_mask=b1["attention_mask"]).last_hidden_state
    l1 = selective_cross_entropy_loss(
        h1, model_ga.lm_head, b1["labels"], weights=b1["weights"], softcap=softcap, weight_sum=total_w
    )
    l1.backward()

    h2 = model_ga.model(input_ids=b2["input_ids"], attention_mask=b2["attention_mask"]).last_hidden_state
    l2 = selective_cross_entropy_loss(
        h2, model_ga.lm_head, b2["labels"], weights=b2["weights"], softcap=softcap, weight_sum=total_w
    )
    l2.backward()

    assert (l1 + l2).item() == pytest.approx(loss_all.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(model_comb.lm_head.weight.grad, model_ga.lm_head.weight.grad, atol=1e-4, rtol=1e-4)
