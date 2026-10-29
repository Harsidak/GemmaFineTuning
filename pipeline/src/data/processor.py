"""Renders agent trajectories into Gemma 4 training examples.

Trajectories are rendered with the tokenizer's own chat template (or an explicit override), never a
hand-rolled format, so training text matches what vLLM renders at serving time. Gemma 4 places tool
responses inside the model turn and TRL ships no Gemma 4 training template, so assistant-only loss
masks are computed here: each assistant span is the text the template adds between the prompt
`messages[:i]` (with generation prompt) and `messages[:i+1]`.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..agent.tools import OFFICIAL_HOST_TOOLS

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class TemplateMismatchError(ValueError):
    """The chat template does not render a trajectory prefix-consistently, so spans are unreliable."""


def default_tools() -> list[dict[str, Any]]:
    """The 9 harness tools in OpenAI function format. Prefer the exact schemas recorded with a trajectory."""
    return [{"type": "function", "function": spec} for spec in OFFICIAL_HOST_TOOLS.values()]


def load_chat_template(path: str | Path | None) -> str | None:
    """Reads a chat_template.jinja override; None keeps the tokenizer's own template."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    logger.info(f"Using chat template override {path} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]})")
    return text


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Maps trajectory messages onto the OpenAI-style schema the Gemma 4 template expects.

    - role "model" becomes "assistant"
    - tool call arguments given as JSON strings are parsed, so the template renders them as Gemma
      key:value arguments instead of pasting the raw string
    - flat tool calls ({"name", "arguments"}) are wrapped as {"type": "function", "function": {...}}
    - text content on a tool-calling message is moved into its reasoning (see below)
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "model":
            msg["role"] = "assistant"
        if msg.get("content") is None:
            msg["content"] = ""
        if msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {"name": tc["name"], "arguments": tc.get("arguments", {})}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        pass  # the template pastes non-JSON strings verbatim
                call = {"type": "function", "function": {"name": fn["name"], "arguments": args}}
                if tc.get("id"):
                    call["id"] = tc["id"]
                calls.append(call)
            msg["tool_calls"] = calls
            # The template renders content *after* the tool responses, but the model generates it before
            # the call, so no single sequence is both the serving context and the generation target.
            # Fold it into the reasoning, which renders before the call.
            if isinstance(msg["content"], str) and msg["content"].strip():
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                msg["reasoning"] = f"{reasoning}\n\n{msg['content'].strip()}".strip()
                msg.pop("reasoning_content", None)
                msg["content"] = ""
        normalized.append(msg)
    return normalized


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template: str | None,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any],
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


def render_with_assistant_spans(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    **template_kwargs: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """Renders a trajectory and returns (text, [(start, end), ...]) character spans of assistant output.

    A span covers everything the model generates for that message: reasoning channel, content, tool
    call blocks and the stop marker (`<|tool_response>` after tool calls, `<turn|>` otherwise).
    """
    msgs = normalize_messages(messages)
    full = _render(tokenizer, msgs, tools, chat_template, False, template_kwargs)

    spans = []
    for i, msg in enumerate(msgs):
        if msg["role"] != "assistant":
            continue
        context = _render(tokenizer, msgs[:i], tools, chat_template, False, template_kwargs)
        prompt = _render(tokenizer, msgs[:i], tools, chat_template, True, template_kwargs)
        upto = _render(tokenizer, msgs[: i + 1], tools, chat_template, False, template_kwargs)
        if not (upto.startswith(context) and full.startswith(upto)):
            raise TemplateMismatchError(f"assistant message {i} does not render as a prefix of the trajectory")
        # The span starts where the generation prompt stops matching. Normally that is its end; a forced
        # opener the message does not render (e.g. `<|channel>thought` with enable_thinking on a turn
        # without reasoning) is left out and the message's own text is trained instead.
        start = _common_prefix_len(prompt, upto)
        if len(upto) > start:
            spans.append((start, len(upto)))
    return full, spans


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def tokenize_trajectory(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    max_length: int | None = None,
    **template_kwargs: Any,
) -> dict[str, list[int]] | None:
    """Returns {"input_ids", "labels"} with loss on assistant tokens only, or None if over max_length.

    Over-long trajectories are dropped rather than truncated: a cut-off agent run teaches neither the
    fix nor submit_patch.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("tokenize_trajectory needs a fast tokenizer (offset mappings)")

    text, spans = render_with_assistant_spans(messages, tokenizer, tools, chat_template, **template_kwargs)
    # The template already emits <bos>; adding special tokens again would double it.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    if max_length is not None and len(input_ids) > max_length:
        return None

    # A token is trained if it overlaps a span; a span can start inside a special token when a forced
    # opener shares its first characters with the message (`<|channel>` vs `<|tool_call>`).
    labels = [IGNORE_INDEX] * len(input_ids)
    span_idx = 0
    for tok_idx, (start, end) in enumerate(enc["offset_mapping"]):
        if end <= start:
            continue
        while span_idx < len(spans) and spans[span_idx][1] <= start:
            span_idx += 1
        if span_idx < len(spans) and spans[span_idx][0] < end:
            labels[tok_idx] = input_ids[tok_idx]
    return {"input_ids": input_ids, "labels": labels}


def format_trajectory_for_sft(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    **template_kwargs: Any,
) -> str:
    """Renders a trajectory to the exact training text (for inspection; training uses tokenize_trajectory)."""
    text, _ = render_with_assistant_spans(messages, tokenizer, tools, chat_template, **template_kwargs)
    return text
