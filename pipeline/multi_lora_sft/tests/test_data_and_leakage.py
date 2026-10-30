"""Unit tests for data contract, role filtering, split, overlong drop, leakage, and cache key (§11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.training.data import (
    DataContractError,
    check_leakage,
    compute_adapter_cache_key,
    deterministic_task_split,
    prepare_adapter_dataset,
)


def _sample_messages(text: str = "Fixed the bug.") -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Fix the bug in app.py"},
        {"role": "assistant", "reasoning": "Inspecting app.py.", "content": text},
    ]


def test_leakage_catches_task_id_repo_commit_and_gold_patch() -> None:
    gold_patch = (
        "--- a/pkg/core.py\n"
        "+++ b/pkg/core.py\n"
        "@@ -10,3 +10,3 @@\n"
        " def run(x):\n"
        "-    return x + 1\n"
        "+    return x + 2\n"
    )
    test_tasks = [
        {
            "instance_id": "repo__pkg-101",
            "repo": "org/pkg",
            "base_commit": "abcdef1234567890",
            "patch": gold_patch,
        }
    ]

    # 1. Clean trajectory on same repo but DIFFERENT commit and DIFFERENT patch -> passes with warning only
    clean_traj = {
        "task_id": "mined_001",
        "repo": "org/pkg",
        "base_commit": "9999999999999999",
        "patch": "--- a/pkg/other.py\n+++ b/pkg/other.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n",
        "messages": _sample_messages(),
    }
    rep_clean = check_leakage([clean_traj], test_tasks)
    assert rep_clean.passed
    assert len(rep_clean.same_repo_warnings) == 1

    # 2. Planted task_id == instance_id -> hard fail
    rep_id = check_leakage([{**clean_traj, "task_id": "repo__pkg-101"}], test_tasks)
    assert not rep_id.passed
    assert rep_id.leaked_task_ids == ["repo__pkg-101"]

    # 3. Planted (repo, base_commit) match -> hard fail
    rep_rc = check_leakage(
        [{**clean_traj, "task_id": "mined_002", "base_commit": "ABCDEF1234567890"}],
        test_tasks,
    )
    assert not rep_rc.passed
    assert len(rep_rc.leaked_repo_commits) == 1

    # 4. Planted identical gold patch with extra `diff --git` header and altered whitespace -> hard fail
    whitespace_varied_patch = (
        "diff --git a/pkg/core.py b/pkg/core.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/pkg/core.py\r\n"
        "+++ b/pkg/core.py\r\n"
        "@@ -10,3 +10,3 @@ def run(x):\r\n"
        "   def run(x):  \r\n"
        "-      return   x + 1\r\n"
        "+      return   x + 2\r\n"
    )
    rep_patch = check_leakage(
        [
            {
                "task_id": "mined_003",
                "repo": "other/fork",
                "base_commit": "111111",
                "submitted_patch": whitespace_varied_patch,
                "messages": _sample_messages(),
            }
        ],
        test_tasks,
    )
    assert not rep_patch.passed
    assert len(rep_patch.leaked_patches) == 1


def test_role_filtering_deterministic_split_overlong_and_cache_key(
    tmp_path: Path, tiny_gemma4_dir: Path, tiny_tokenizer: Any
) -> None:
    cfg = load_config(
        set_overrides=[
            f"model.served_path={tiny_gemma4_dir.as_posix()}",
            "data.val_fraction=0.25",
            (
                "adapters=[{name: main_lora, roles: [root], overrides: {train: {max_seq_length: 6000}}},"
                " {name: tool_lora, roles: [code_analyzer], overrides: {train: {max_seq_length: 6000}}}]"
            ),
        ]
    )

    records: list[dict[str, Any]] = []
    for i in range(8):
        records.append(
            {
                "task_id": f"task_{i:02d}",
                "agent": "root",
                "resolved": True,
                "messages": _sample_messages(f"Root fix #{i}"),
            }
        )
    # 1 analyzer record + 1 over-long root record (> 6000 byte-level tokens)
    records.append(
        {
            "task_id": "task_analyzer_01",
            "agent": "code_analyzer",
            "resolved": True,
            "messages": _sample_messages("Analyzer report."),
        }
    )
    records.append(
        {
            "task_id": "task_overlong",
            "agent": "root",
            "resolved": True,
            "messages": _sample_messages("word " * 1200),
        }
    )

    eff_main = cfg.resolve_adapter("main_lora")
    summary_main = prepare_adapter_dataset(
        cfg=cfg,
        eff_adapter=eff_main,
        raw_records=records,
        data_file_hashes={"train.jsonl": "abc123"},
        tokenizer=tiny_tokenizer,
        served_tokenizer_path=str(tiny_gemma4_dir),
        chat_template=tiny_tokenizer.chat_template,
        template_sha256="tpl_hash_v1",
        tokenizer_sha256="tok_hash_v1",
        out_cache_dir=tmp_path / "cache" / "main_lora",
    )

    assert summary_main["dropped_role_filter"] == 1
    assert summary_main["dropped_overlong"] == 1
    assert summary_main["train_examples"] + summary_main["val_examples"] == 8
    assert summary_main["val_examples"] >= 1

    # Verify deterministic task_id split (disjoint task_ids between train and val across repeated calls)
    fake_items = [{"task_id": f"t_{i // 2}", "text_sha256": f"h_{i}"} for i in range(20)]
    tr_a, va_a = deterministic_task_split(fake_items, val_fraction=0.2, val_max_examples=64, seed=1234)
    tr_b, va_b = deterministic_task_split(fake_items, val_fraction=0.2, val_max_examples=64, seed=1234)
    assert tr_a == tr_b and va_a == va_b
    assert {r["task_id"] for r in tr_a}.isdisjoint({r["task_id"] for r in va_a})

    # Verify missing `agent` field fails when role filter is non-null and 0 match
    with pytest.raises(DataContractError, match="missing the required 'agent' field"):
        prepare_adapter_dataset(
            cfg=cfg,
            eff_adapter=eff_main,
            raw_records=[{"task_id": "no_agent", "resolved": True, "messages": _sample_messages()}],
            data_file_hashes={"train.jsonl": "abc"},
            tokenizer=tiny_tokenizer,
            served_tokenizer_path=str(tiny_gemma4_dir),
            chat_template=tiny_tokenizer.chat_template,
            template_sha256="tpl_hash_v1",
            tokenizer_sha256="tok_hash_v1",
            out_cache_dir=tmp_path / "cache_fail",
        )

    # Verify cache key changes whenever chat template sha256 changes
    key_v1 = compute_adapter_cache_key(
        data_file_hashes={"train.jsonl": "abc123"},
        template_sha256="tpl_hash_v1",
        tokenizer_sha256="tok_hash_v1",
        eff_adapter=eff_main,
        successful_only=True,
        val_fraction=0.25,
        val_max_examples=64,
        template_kwargs={},
        seed=1234,
    )
    key_v2 = compute_adapter_cache_key(
        data_file_hashes={"train.jsonl": "abc123"},
        template_sha256="tpl_hash_v2_changed",
        tokenizer_sha256="tok_hash_v1",
        eff_adapter=eff_main,
        successful_only=True,
        val_fraction=0.25,
        val_max_examples=64,
        template_kwargs={},
        seed=1234,
    )
    assert key_v1 != key_v2
    assert json.loads((tmp_path / "cache" / "main_lora" / "cache_meta.json").read_text(encoding="utf-8"))["cache_key"] == key_v1
