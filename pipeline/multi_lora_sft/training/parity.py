"""HF-side adapter parity reference generation and vLLM vs HF parity comparison.

Implements design_doc.md §7.10 and §9.3:
- Extracts fixed assistant-turn context prefixes (<= `max_prompt_tokens`, default 4096) from the
  validation set (falling back to train set when val is empty).
- Decodes greedily for `gen_tokens` tokens both WITH the LoRA adapter and WITHOUT (`disable_adapter()`),
  recording each chosen token's log-prob and top-5 log-probs.
- Compares vLLM greedy token sequences and log-probs against the HF reference to detect int4 + LoRA
  serving regressions (e.g. vllm#50059).
- Under FSDP the decode runs on every rank in lockstep through the sharded model (each forward all-gathers the
  layers); rank 0 picks each token and broadcasts it so all ranks feed the same next input and stop together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ...src.data.processor import IGNORE_INDEX
from .distributed import broadcast_int, is_fsdp_wrapped

logger = logging.getLogger(__name__)


def extract_parity_prompts(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    num_prompts: int = 8,
    max_prompt_tokens: int = 4096,
) -> list[dict[str, Any]]:
    """Extracts up to `num_prompts` assistant-turn context prefixes capped at `max_prompt_tokens`."""
    prompts: list[dict[str, Any]] = []
    for rec in records:
        if len(prompts) >= num_prompts:
            break
        ids = list(rec["input_ids"])
        labels = list(rec["labels"])
        # Find the start of the first supervised assistant span
        first_sup = next((i for i, y in enumerate(labels) if y != IGNORE_INDEX and i > 0), None)
        if first_sup is None:
            continue
        prefix_ids = ids[:first_sup]
        if len(prefix_ids) > max_prompt_tokens:
            # Prefer shorter turns if available; otherwise keep last max_prompt_tokens
            prefix_ids = prefix_ids[-max_prompt_tokens:]
        prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        prompts.append(
            {
                "task_id": str(rec.get("task_id", f"prompt_{len(prompts)}")),
                "prompt_token_ids": prefix_ids,
                "prompt_text": prompt_text,
            }
        )
    return prompts


@torch.no_grad()
def greedy_decode_with_logprobs(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    gen_tokens: int = 64,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Runs exact step-by-step greedy decoding on `model`, recording chosen and top-k logprobs."""
    was_training = model.training
    model.eval()
    try:
        # An FSDP root must run its own forward (it un-shards the embeddings / lm_head it holds); calling `.module`
        # would read freed shards. Other wrappers (DDP) are bypassed as before.
        lockstep = is_fsdp_wrapped(model)
        unwrapped = model if lockstep else (model.module if hasattr(model, "module") else model)
        dev = next(unwrapped.parameters()).device
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        steps: list[dict[str, Any]] = []
        eos_id = tokenizer.eos_token_id

        for _ in range(max(1, int(gen_tokens))):
            out = unwrapped(input_ids=cur_ids, use_cache=False, logits_to_keep=1)
            next_logits = out.logits[0, -1, :].float()
            logprobs = F.log_softmax(next_logits, dim=-1)

            chosen_id = int(torch.argmax(logprobs).item())
            if lockstep:
                chosen_id = broadcast_int(chosen_id)
            chosen_lp = float(logprobs[chosen_id].item())
            k = min(top_k, logprobs.numel())
            top_vals, top_idxs = torch.topk(logprobs, k=k)
            top_list = [
                {
                    "token_id": int(tid.item()),
                    "token": tokenizer.decode([int(tid.item())], skip_special_tokens=False),
                    "logprob": round(float(val.item()), 6),
                }
                for val, tid in zip(top_vals, top_idxs, strict=True)
            ]

            steps.append(
                {
                    "token_id": chosen_id,
                    "token": tokenizer.decode([chosen_id], skip_special_tokens=False),
                    "logprob": round(chosen_lp, 6),
                    "top5": top_list,
                }
            )
            next_t = torch.tensor([[chosen_id]], dtype=torch.long, device=dev)
            cur_ids = torch.cat([cur_ids, next_t], dim=1)
            if eos_id is not None and chosen_id == eos_id:
                break
        return steps
    finally:
        if was_training:
            model.train()


def generate_hf_parity_reference(
    *,
    adapter_name: str,
    peft_model: Any,
    tokenizer: Any,
    val_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    output_path: Path,
    num_prompts: int = 8,
    max_prompt_tokens: int = 4096,
    gen_tokens: int = 64,
    write: bool = True,
) -> Path:
    """Generates HF greedy tokens + logprobs with and without the adapter and saves JSON (§7.10).

    `peft_model` may be FSDP-wrapped, in which case every rank must call this together and only the rank with
    `write=True` writes the file."""
    source_records = val_records if val_records else train_records
    prompts = extract_parity_prompts(
        source_records,
        tokenizer,
        num_prompts=num_prompts,
        max_prompt_tokens=max_prompt_tokens,
    )

    unwrapped = peft_model.module if hasattr(peft_model, "module") else peft_model  # the PeftModel
    runner = peft_model if is_fsdp_wrapped(peft_model) else unwrapped  # what the forward is called on
    items: list[dict[str, Any]] = []
    for p in prompts:
        with_adapter = greedy_decode_with_logprobs(
            runner, tokenizer, p["prompt_token_ids"], gen_tokens=gen_tokens
        )
        with unwrapped.disable_adapter():
            without_adapter = greedy_decode_with_logprobs(
                runner, tokenizer, p["prompt_token_ids"], gen_tokens=gen_tokens
            )
        items.append(
            {
                "task_id": p["task_id"],
                "prompt_text": p["prompt_text"],
                "prompt_len_tokens": len(p["prompt_token_ids"]),
                "with_adapter": with_adapter,
                "without_adapter": without_adapter,
            }
        )

    payload = {
        "adapter": adapter_name,
        "num_prompts": len(items),
        "gen_tokens": gen_tokens,
        "prompts": items,
    }
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _score_single_arm(
    hf_sequences: list[list[dict[str, Any]]],
    vllm_sequences: list[list[dict[str, Any]]],
) -> tuple[float, float | None]:
    """Computes greedy agreement (up to first divergence per prompt) and mean |Δ log-prob| (§9.3)."""
    matched_positions = 0
    compared_positions = 0
    abs_lp_diffs: list[float] = []

    for hf_steps, vllm_steps in zip(hf_sequences, vllm_sequences, strict=False):
        n = min(len(hf_steps), len(vllm_steps))
        if n == 0:
            continue
        for pos in range(n):
            hf_tok = hf_steps[pos].get("token")
            hf_id = hf_steps[pos].get("token_id")
            v_step = vllm_steps[pos]
            v_tok = v_step.get("token")
            v_id = v_step.get("token_id")

            # Record |Δ log-prob| of HF-chosen token if reported in vLLM top-logprobs
            v_top = v_step.get("top_logprobs") or {}
            hf_lp = hf_steps[pos].get("logprob")
            if hf_lp is not None:
                if hf_tok in v_top:
                    abs_lp_diffs.append(abs(float(hf_lp) - float(v_top[hf_tok])))
                elif v_tok == hf_tok and v_step.get("logprob") is not None:
                    abs_lp_diffs.append(abs(float(hf_lp) - float(v_step["logprob"])))

            compared_positions += 1
            same = (hf_id is not None and v_id is not None and int(hf_id) == int(v_id)) or (hf_tok == v_tok)
            if same:
                matched_positions += 1
            else:
                # Per §9.3: fraction of positions *up to the first divergence* across all prompts
                break

    agreement = matched_positions / max(1, compared_positions)
    mean_dlp = (sum(abs_lp_diffs) / len(abs_lp_diffs)) if abs_lp_diffs else None
    return round(agreement, 4), (round(mean_dlp, 4) if mean_dlp is not None else None)


def evaluate_parity_against_hf(
    hf_ref: dict[str, Any],
    vllm_with_adapter: list[list[dict[str, Any]]],
    vllm_without_adapter: list[list[dict[str, Any]]],
    *,
    max_agreement_drop: float = 0.10,
) -> dict[str, Any]:
    """Computes `OK` vs `SUSPECT` parity verdict comparing adapter agreement against base agreement (§9.3)."""
    prompts = hf_ref.get("prompts", [])
    hf_with = [p["with_adapter"] for p in prompts]
    hf_base = [p["without_adapter"] for p in prompts]

    adapter_agree, adapter_dlp = _score_single_arm(hf_with, vllm_with_adapter)
    base_agree, base_dlp = _score_single_arm(hf_base, vllm_without_adapter)

    verdict = "OK" if adapter_agree >= (base_agree - max_agreement_drop - 1e-9) else "SUSPECT"
    recommendation = (
        None
        if verdict == "OK"
        else "retrain at r=16 or with fewer target modules (vllm#50059)"
    )
    return {
        "adapter": hf_ref.get("adapter"),
        "verdict": verdict,
        "adapter_greedy_agreement": adapter_agree,
        "base_greedy_agreement": base_agree,
        "agreement_drop": round(base_agree - adapter_agree, 4),
        "max_agreement_drop": max_agreement_drop,
        "adapter_mean_abs_delta_logprob": adapter_dlp,
        "base_mean_abs_delta_logprob": base_dlp,
        "recommendation": recommendation,
    }
