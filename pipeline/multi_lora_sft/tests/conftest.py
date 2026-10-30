"""Shared pytest fixtures for fast offline CPU testing of `pipeline.multi_lora_sft`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.multi_lora_sft.training.model import save_tiny_gemma4_checkpoint


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def tiny_gemma4_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materializes a tiny 4-layer Gemma 4 checkpoint (with vision + KV sharing + softcap) once per session."""
    base = tmp_path_factory.mktemp("tiny_gemma4")
    return save_tiny_gemma4_checkpoint(
        base,
        vocab_size=512,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        include_vision=True,
        seed=42,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_gemma4_dir: Path) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_gemma4_dir), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
