"""Data contract validation, leakage detection, role filtering, tokenization, cache, and sampler.

Implements design_doc.md §7.1, §7.2, §7.5, and §11, reusing `pipeline.src.data.processor`,
`pipeline.src.data.loader`, `pipeline.src.agent.tools`, and `pipeline.src.utils.packager`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...src.agent.tools import OFFICIAL_HOST_TOOLS
from ...src.data.loader import load_tasks
from ...src.data.processor import (
    IGNORE_INDEX,
    TemplateMismatchError,
    default_tools,
    load_chat_template,
    render_with_assistant_spans,
    tokenize_trajectory,
)
from ...src.utils.packager import ADKYamlLoader
from ..config import EffectiveAdapterConfig, PipelineConfig
from ..runctx import sha256_bytes, sha256_file

logger = logging.getLogger(__name__)


class DataContractError(ValueError):
    """Raised on hard preflight failures (leakage, broken schema, missing template, etc.)."""


@dataclass
class LeakageReport:
    passed: bool
    leaked_task_ids: list[str] = field(default_factory=list)
    leaked_repo_commits: list[str] = field(default_factory=list)
    leaked_patches: list[str] = field(default_factory=list)
    invalid_eval_task_ids: list[str] = field(default_factory=list)
    same_repo_warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "leaked_task_ids": self.leaked_task_ids,
            "leaked_repo_commits": self.leaked_repo_commits,
            "leaked_patches": self.leaked_patches,
            "invalid_eval_task_ids": self.invalid_eval_task_ids,
            "same_repo_warnings": self.same_repo_warnings,
        }


def normalize_patch_diff(patch: str) -> str:
    """Normalizes a unified git diff for whitespace-insensitive leakage matching (§11).

    Strips `diff --git` / `index` / mode metadata lines so patches starting directly with
    `--- a/...` (as in `tasks.jsonl`) match full `git diff` outputs identically.
    """
    if not patch or not patch.strip():
        return ""
    norm_lines: list[str] = []
    for raw_line in patch.replace("\r\n", "\n").splitlines():
        s = raw_line.strip()
        if not s:
            continue
        if s.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ", "similarity index ")):
            continue
        if s.startswith(("--- ", "+++ ")):
            prefix = s[:4]
            target = s[4:].strip()
            if target.startswith(("a/", "b/")):
                target = target[2:]
            norm_lines.append(f"{prefix}{target}")
            continue
        if s.startswith("@@"):
            # Skip unified diff hunk header lines (`@@ -a,b +c,d @@ [optional context]`) so
            # git xfuncname header annotations or shifted line counts never mask identical hunks.
            continue
        if raw_line[0] in {"+", "-"}:
            sign = raw_line[0]
            body = re.sub(r"\s+", " ", raw_line[1:].strip())
            norm_lines.append(f"{sign}{body}")
        else:
            norm_lines.append(" " + re.sub(r"\s+", " ", s))
    return "\n".join(norm_lines)


def extract_trajectory_patches(traj: dict[str, Any]) -> list[str]:
    """Extracts candidate final unified diffs from a trajectory record and its messages."""
    patches: list[str] = []
    for key in ("patch", "submitted_patch", "agent_patch", "git_diff", "diff"):
        val = traj.get(key)
        if isinstance(val, str) and val.strip():
            patches.append(val)

    messages = traj.get("messages") or []
    edit_diffs: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        # Check tool call arguments (e.g., submit_patch({"patch": ...}))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or tc
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = None
            if isinstance(args, dict):
                for k in ("patch", "diff"):
                    if isinstance(args.get(k), str) and args[k].strip():
                        patches.append(args[k])

        content = msg.get("content")
        if isinstance(content, str) and ("--- " in content and "+++ " in content):
            # Could be JSON tool output (`{"diff": "..."}`) or raw unified diff
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                parsed = json.loads(content)
                if isinstance(parsed, dict) and isinstance(parsed.get("diff"), str):
                    edit_diffs.append(parsed["diff"])
                    continue
            patches.append(content)
        elif isinstance(content, str) and content.strip().startswith("{"):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for k in ("patch", "submitted_patch", "diff"):
                        if isinstance(parsed.get(k), str) and parsed[k].strip():
                            edit_diffs.append(parsed[k])

    if edit_diffs:
        patches.extend(edit_diffs)
        if len(edit_diffs) > 1:
            patches.append("\n".join(reversed(edit_diffs)))
    return patches


def check_leakage(
    trajectories: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
    eval_task_ids: list[str] | None = None,
) -> LeakageReport:
    """Checks training trajectories against held-out test tasks per design_doc.md §11."""
    test_ids: set[str] = set()
    test_repo_commits: dict[tuple[str, str], str] = {}
    test_patch_hashes: dict[str, str] = {}
    test_repos: dict[str, list[str]] = {}

    for t in test_tasks:
        tid = str(t.get("instance_id") or t.get("task_id") or "").strip()
        if tid:
            test_ids.add(tid)
        repo = str(t.get("repo") or "").strip().lower()
        commit = str(t.get("base_commit") or "").strip().lower()
        if repo:
            test_repos.setdefault(repo, []).append(commit)
        if repo and commit:
            test_repo_commits[(repo, commit)] = tid or f"{repo}@{commit[:12]}"
        gold_patch = str(t.get("patch") or "")
        norm_gold = normalize_patch_diff(gold_patch)
        if norm_gold:
            h = sha256_bytes(norm_gold.encode("utf-8"))
            test_patch_hashes[h] = tid or "unknown_test_task"

    leaked_ids: set[str] = set()
    leaked_rc: set[str] = set()
    leaked_patches: set[str] = set()
    same_repo_warnings: list[dict[str, Any]] = []
    train_ids: set[str] = set()

    for idx, traj in enumerate(trajectories):
        tid = str(traj.get("task_id") or traj.get("instance_id") or "").strip()
        if tid:
            train_ids.add(tid)
        if tid and tid in test_ids:
            leaked_ids.add(tid)

        repo = str(traj.get("repo") or "").strip().lower()
        commit = str(traj.get("base_commit") or "").strip().lower()
        if repo and commit and (repo, commit) in test_repo_commits:
            leaked_rc.add(f"{tid or f'row_{idx}'}=={test_repo_commits[(repo, commit)]}({repo}@{commit[:12]})")
        elif repo and repo in test_repos:
            same_repo_warnings.append(
                {
                    "task_id": tid or f"row_{idx}",
                    "repo": repo,
                    "base_commit": commit or None,
                    "test_task_commits_in_repo": len(test_repos[repo]),
                }
            )

        for cand_patch in extract_trajectory_patches(traj):
            norm_cand = normalize_patch_diff(cand_patch)
            if norm_cand:
                ch = sha256_bytes(norm_cand.encode("utf-8"))
                if ch in test_patch_hashes:
                    leaked_patches.add(f"{tid or f'row_{idx}'}=={test_patch_hashes[ch]}")

    invalid_eval: list[str] = []
    if eval_task_ids:
        for eid in eval_task_ids:
            if test_ids and eid not in test_ids:
                invalid_eval.append(f"{eid} (missing from tasks.jsonl)")
            if eid in train_ids:
                invalid_eval.append(f"{eid} (present in training data)")

    passed = not (leaked_ids or leaked_rc or leaked_patches or invalid_eval)
    return LeakageReport(
        passed=passed,
        leaked_task_ids=sorted(leaked_ids),
        leaked_repo_commits=sorted(leaked_rc),
        leaked_patches=sorted(leaked_patches),
        invalid_eval_task_ids=invalid_eval,
        same_repo_warnings=same_repo_warnings[:100],
    )


def resolve_role_tools(
    role: str | None,
    template_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Resolves the OpenAI-format tool schemas for `role` from `template_dir`, else `default_tools()`."""
    if template_dir is not None:
        tdir = Path(template_dir)
        yaml_file: Path | None = None
        if role == "code_analyzer" and (tdir / "sub_agents" / "code_analyzer.yaml").exists():
            yaml_file = tdir / "sub_agents" / "code_analyzer.yaml"
        elif (tdir / "agent.yaml").exists():
            yaml_file = tdir / "agent.yaml"

        if yaml_file is not None:
            try:
                import yaml

                parsed = yaml.load(yaml_file.read_text(encoding="utf-8"), Loader=ADKYamlLoader)
                if isinstance(parsed, dict) and isinstance(parsed.get("tools"), list):
                    resolved: list[dict[str, Any]] = []
                    for entry in parsed["tools"]:
                        if isinstance(entry, str) and entry in OFFICIAL_HOST_TOOLS:
                            resolved.append({"type": "function", "function": OFFICIAL_HOST_TOOLS[entry]})
                    if resolved:
                        return resolved
            except (OSError, ValueError) as exc:
                logger.warning("Failed reading tools from %s: %s; using default_tools()", yaml_file, exc)

    if role == "code_analyzer":
        analyzer_names = ("read_file", "search_similar_code", "get_code_neighbors", "get_code_subgraph")
        return [
            {"type": "function", "function": OFFICIAL_HOST_TOOLS[n]}
            for n in analyzer_names
            if n in OFFICIAL_HOST_TOOLS
        ]
    return default_tools()


def load_served_tokenizer_and_template(
    cfg: PipelineConfig,
    *,
    tiny_fallback_dir: Path | None = None,
) -> tuple[Any, str, str, str]:
    """Loads fast tokenizer and chat template from `model.served_path`, returning `(tok, tpl, tpl_sha, tok_sha)`."""
    from transformers import AutoTokenizer

    from .model import save_tiny_gemma4_checkpoint

    served_path = cfg.model.served_path or cfg.model.train_base_path
    if served_path in {None, "__tiny__"}:
        if tiny_fallback_dir is None:
            raise DataContractError("Neither model.served_path nor model.train_base_path is set")
        served_path = str(save_tiny_gemma4_checkpoint(tiny_fallback_dir))

    spath = Path(served_path)
    tokenizer = AutoTokenizer.from_pretrained(served_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    jinja_file = spath / "chat_template.jinja" if spath.is_dir() else None
    if jinja_file is not None and jinja_file.is_file():
        chat_template = load_chat_template(jinja_file) or ""
    else:
        chat_template = getattr(tokenizer, "chat_template", None) or ""

    if not chat_template:
        raise DataContractError(f"No chat template found in served model path: {served_path}")

    tpl_sha = sha256_bytes(chat_template.encode("utf-8"))
    expected = cfg.model.expected_template_sha256
    if expected:
        exp_norm = expected.strip().lower()
        if not (tpl_sha == exp_norm or (len(exp_norm) >= 12 and tpl_sha.startswith(exp_norm))):
            raise DataContractError(
                f"Chat template sha256 mismatch: got {tpl_sha}, expected {expected!r}"
            )

    if spath.is_dir() and (spath / "tokenizer.json").is_file():
        tok_sha = sha256_file(spath / "tokenizer.json")
    else:
        vocab_items = sorted(tokenizer.get_vocab().items())
        tok_sha = sha256_bytes(json.dumps(vocab_items).encode("utf-8"))

    return tokenizer, chat_template, tpl_sha, tok_sha


def verify_tool_call_wire_format(rendered_text: str, messages: list[dict[str, Any]]) -> bool:
    """Checks that every assistant tool call rendered cleanly into `<|tool_call>call:name{...}<tool_call|>`."""
    expected_names: list[str] = []
    for msg in messages:
        if msg.get("role") in {"assistant", "model"} and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or tc
                if fn.get("name"):
                    expected_names.append(str(fn["name"]))

    open_cnt = rendered_text.count("<|tool_call>")
    close_cnt = rendered_text.count("<tool_call|>")
    if open_cnt != close_cnt or open_cnt < len(expected_names):
        return False

    found_calls = re.findall(r"<\|tool_call>call:([A-Za-z0-9_.-]+)\{", rendered_text)
    return found_calls == expected_names


def compute_token_weights(
    input_ids: list[int],
    labels: list[int],
    tokenizer: Any,
    tool_call_weight: float,
) -> list[float]:
    """Assigns `tool_call_weight` to supervised tokens in `<|tool_call> ... <tool_call|>` inclusive (§7.2)."""
    start_id = tokenizer.convert_tokens_to_ids("<|tool_call>")
    end_id = tokenizer.convert_tokens_to_ids("<tool_call|>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        start_id is None
        or end_id is None
        or (unk_id is not None and (start_id == unk_id or end_id == unk_id))
    ):
        raise DataContractError(
            "Tokenizer is missing special tokens '<|tool_call>' or '<tool_call|>' needed for tool_call_weight"
        )

    weights: list[float] = []
    in_call = False
    for tok_id, lbl in zip(input_ids, labels, strict=True):
        if tok_id == start_id:
            in_call = True
        if lbl == IGNORE_INDEX:
            weights.append(0.0)
        elif in_call:
            weights.append(float(tool_call_weight))
        else:
            weights.append(1.0)
        if tok_id == end_id:
            in_call = False
    return weights


def deterministic_task_split(
    records: list[dict[str, Any]],
    *,
    val_fraction: float,
    val_max_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically holds out ~`val_fraction` of unique `task_id`s, capped at `val_max_examples`."""
    if not records or val_fraction <= 0.0 or val_max_examples <= 0:
        return list(records), []

    unique_tasks = sorted({str(r["task_id"]) for r in records})
    if len(unique_tasks) <= 1:
        # Never leave train empty when there is only a single unique task
        return list(records), []

    val_tasks: set[str] = set()
    scored_tasks: list[tuple[float, str]] = []
    for tid in unique_tasks:
        digest = hashlib.sha256(f"{seed}:{tid}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        scored_tasks.append((u, tid))
        if u < val_fraction:
            val_tasks.add(tid)

    # Guarantee at least 1 val task when dataset has >= 20 tasks and val_fraction > 0, while keeping >= 1 train task
    if not val_tasks and len(unique_tasks) >= max(2, round(1.0 / val_fraction)):
        scored_tasks.sort(key=lambda x: x[0])
        val_tasks.add(scored_tasks[0][1])
    if len(val_tasks) == len(unique_tasks):
        scored_tasks.sort(key=lambda x: x[0])
        val_tasks.discard(scored_tasks[-1][1])

    train_recs: list[dict[str, Any]] = []
    val_recs: list[dict[str, Any]] = []
    for r in sorted(records, key=lambda d: (str(d["task_id"]), d.get("text_sha256", ""))):
        if str(r["task_id"]) in val_tasks and len(val_recs) < val_max_examples:
            val_recs.append(r)
        else:
            train_recs.append(r)

    return train_recs, val_recs


def compute_adapter_cache_key(
    *,
    data_file_hashes: dict[str, str],
    template_sha256: str,
    tokenizer_sha256: str,
    eff_adapter: EffectiveAdapterConfig,
    successful_only: bool,
    val_fraction: float,
    val_max_examples: int,
    template_kwargs: dict[str, Any],
    seed: int,
) -> str:
    payload = {
        "data_files": sorted(data_file_hashes.items()),
        "template_sha256": template_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "max_seq_length": eff_adapter.train.max_seq_length,
        "template_kwargs": template_kwargs,
        "roles": sorted(eff_adapter.roles) if eff_adapter.roles is not None else None,
        "successful_only": successful_only,
        "val_fraction": val_fraction,
        "val_max_examples": val_max_examples,
        "tool_call_weight": eff_adapter.loss.tool_call_weight,
        "seed": seed,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))[:24]


def _tokenize_worker_item(args: tuple[Any, ...]) -> dict[str, Any]:
    """Worker callable for `ProcessPoolExecutor` or sequential loop."""
    (
        idx,
        task_id,
        messages,
        tools,
        chat_template,
        max_seq_length,
        template_kwargs,
        tool_call_weight,
    ) = args
    tok = _WORKER_TOKENIZER
    try:
        rendered, _ = render_with_assistant_spans(
            messages, tok, tools=tools, chat_template=chat_template, **template_kwargs
        )
        wire_ok = verify_tool_call_wire_format(rendered, messages)
        rec = tokenize_trajectory(
            messages,
            tok,
            tools=tools,
            chat_template=chat_template,
            max_length=max_seq_length,
            **template_kwargs,
        )
    except TemplateMismatchError as exc:
        return {"status": "template_mismatch", "idx": idx, "error": str(exc)}

    if rec is None:
        return {"status": "overlong", "idx": idx, "wire_ok": wire_ok}

    labels = rec["labels"]
    sup_tokens = sum(1 for y in labels[1:] if y != IGNORE_INDEX)
    if sup_tokens == 0:
        return {"status": "no_assistant_tokens", "idx": idx, "wire_ok": wire_ok}

    weights = (
        compute_token_weights(rec["input_ids"], labels, tok, tool_call_weight)
        if abs(tool_call_weight - 1.0) > 1e-9
        else None
    )
    text_sha = sha256_bytes(rendered.encode("utf-8"))
    return {
        "status": "ok",
        "idx": idx,
        "task_id": task_id,
        "input_ids": rec["input_ids"],
        "labels": labels,
        "weights": weights,
        "length": len(rec["input_ids"]),
        "supervised_tokens": sup_tokens,
        "text_sha256": text_sha,
        "wire_ok": wire_ok,
    }


_WORKER_TOKENIZER: Any = None


def _init_tokenize_worker(served_path: str) -> None:
    global _WORKER_TOKENIZER
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(served_path, trust_remote_code=True, use_fast=True)
    if _WORKER_TOKENIZER.pad_token is None:
        _WORKER_TOKENIZER.pad_token = _WORKER_TOKENIZER.eos_token


def save_parquet_split(records: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    has_weights = any(r.get("weights") is not None for r in records)
    fields = [
        pa.field("task_id", pa.string()),
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("labels", pa.list_(pa.int32())),
        pa.field("length", pa.int32()),
        pa.field("supervised_tokens", pa.int32()),
    ]
    if has_weights:
        fields.append(pa.field("weights", pa.list_(pa.float32())))
    schema = pa.schema(fields)

    cols: dict[str, list[Any]] = {
        "task_id": [str(r["task_id"]) for r in records],
        "input_ids": [r["input_ids"] for r in records],
        "labels": [r["labels"] for r in records],
        "length": [int(r["length"]) for r in records],
        "supervised_tokens": [int(r["supervised_tokens"]) for r in records],
    }
    if has_weights:
        cols["weights"] = [r["weights"] for r in records]

    table = pa.Table.from_pydict(cols, schema=schema)
    pq.write_table(table, path)


def load_parquet_split(path: Path, max_seq_length: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    table = pq.read_table(path)
    rows = table.to_pylist()
    if max_seq_length is not None:
        rows = [r for r in rows if int(r["length"]) <= max_seq_length]
    return rows


def load_raw_trajectories_checked(train_files: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Loads raw JSONL records from `train_files`, validating basic trajectory schema."""
    if not train_files:
        raise DataContractError("data.train_files is empty; specify at least one JSONL file")

    records: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for fp_str in train_files:
        p = Path(fp_str)
        if not p.is_file():
            raise DataContractError(f"Training data file not found: {p}")
        file_hashes[str(p)] = sha256_file(p)
        with open(p, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                s = raw_line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as exc:
                    raise DataContractError(f"Invalid JSON in {p}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise DataContractError(f"Record in {p}:{line_no} is not a JSON object")
                tid = obj.get("task_id") or obj.get("instance_id")
                if not tid or not isinstance(tid, str):
                    raise DataContractError(f"Record in {p}:{line_no} missing string 'task_id'")
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    raise DataContractError(f"Record {tid!r} in {p}:{line_no} missing non-empty 'messages' list")
                for m_idx, m in enumerate(msgs):
                    if not isinstance(m, dict) or m.get("role") not in {
                        "user",
                        "assistant",
                        "model",
                        "tool",
                        "system",
                        "developer",
                    }:
                        raise DataContractError(
                            f"Record {tid!r} in {p}:{line_no} message[{m_idx}] has invalid role"
                        )
                records.append(obj)

    if not records:
        raise DataContractError("All training files were empty")
    return records, file_hashes


def prepare_adapter_dataset(
    *,
    cfg: PipelineConfig,
    eff_adapter: EffectiveAdapterConfig,
    raw_records: list[dict[str, Any]],
    data_file_hashes: dict[str, str],
    tokenizer: Any,
    served_tokenizer_path: str | None,
    chat_template: str,
    template_sha256: str,
    tokenizer_sha256: str,
    out_cache_dir: Path,
) -> dict[str, Any]:
    """Filters, tokenizes, weights, splits, and caches trajectories for one adapter (§7.2)."""
    cache_key = compute_adapter_cache_key(
        data_file_hashes=data_file_hashes,
        template_sha256=template_sha256,
        tokenizer_sha256=tokenizer_sha256,
        eff_adapter=eff_adapter,
        successful_only=cfg.data.successful_only,
        val_fraction=cfg.data.val_fraction,
        val_max_examples=cfg.data.val_max_examples,
        template_kwargs=cfg.data.template_kwargs,
        seed=cfg.run.seed,
    )

    out_cache_dir.mkdir(parents=True, exist_ok=True)
    # Reuse attached pre-tokenized cache if key matches (§7.2 step 5)
    if cfg.data.cache_dir:
        ext_dir = Path(cfg.data.cache_dir) / eff_adapter.name
        ext_meta_path = ext_dir / "cache_meta.json"
        if ext_meta_path.is_file():
            ext_meta = json.loads(ext_meta_path.read_text(encoding="utf-8"))
            if ext_meta.get("cache_key") == cache_key and (ext_dir / "train.parquet").is_file():
                import shutil

                for fn in ("train.parquet", "val.parquet", "cache_meta.json"):
                    if (ext_dir / fn).is_file():
                        shutil.copy2(ext_dir / fn, out_cache_dir / fn)
                ext_meta["reused_external_cache"] = str(ext_dir)
                return ext_meta

    local_meta = out_cache_dir / "cache_meta.json"
    if local_meta.is_file():
        meta = json.loads(local_meta.read_text(encoding="utf-8"))
        if meta.get("cache_key") == cache_key and (out_cache_dir / "train.parquet").is_file():
            meta["reused_local_cache"] = True
            return meta

    roles_set = set(eff_adapter.roles) if eff_adapter.roles else None
    missing_agent_count = 0
    dropped_role = 0
    dropped_unresolved = 0
    filtered: list[dict[str, Any]] = []

    for r in raw_records:
        if cfg.data.successful_only and not r.get("resolved", True):
            dropped_unresolved += 1
            continue
        if roles_set is not None:
            agent_role = r.get("agent")
            if not agent_role:
                missing_agent_count += 1
                continue
            if str(agent_role) not in roles_set:
                dropped_role += 1
                continue
        filtered.append(r)

    if roles_set is not None and missing_agent_count > 0 and not filtered:
        raise DataContractError(
            f"Adapter {eff_adapter.name!r} filters roles={sorted(roles_set)}, but {missing_agent_count} "
            "record(s) are missing the required 'agent' field and 0 matched."
        )
    if not filtered:
        raise DataContractError(
            f"Adapter {eff_adapter.name!r} has 0 matching records after filtering "
            f"(roles={eff_adapter.roles}, dropped_unresolved={dropped_unresolved}, dropped_role={dropped_role})"
        )

    work_items = []
    for idx, r in enumerate(filtered):
        tid = str(r.get("task_id") or r.get("instance_id"))
        role = r.get("agent")
        tools = r.get("tools") or resolve_role_tools(role, cfg.submission.template_dir)
        work_items.append(
            (
                idx,
                tid,
                r["messages"],
                tools,
                chat_template,
                eff_adapter.train.max_seq_length,
                cfg.data.template_kwargs,
                eff_adapter.loss.tool_call_weight,
            )
        )

    n_workers = int(cfg.data.num_workers)
    if n_workers > 1 and len(work_items) >= 32 and served_tokenizer_path and Path(served_tokenizer_path).exists():
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_tokenize_worker,
            initargs=(served_tokenizer_path,),
        ) as pool:
            results = list(pool.map(_tokenize_worker_item, work_items))
    else:
        global _WORKER_TOKENIZER
        _WORKER_TOKENIZER = tokenizer
        results = [_tokenize_worker_item(item) for item in work_items]

    kept_dedup: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    dropped_overlong = 0
    dropped_mismatch = 0
    dropped_no_targets = 0
    dropped_duplicates = 0
    wire_format_warnings = 0

    for res in results:
        st = res["status"]
        if not res.get("wire_ok", True):
            wire_format_warnings += 1
        if st == "template_mismatch":
            dropped_mismatch += 1
        elif st == "overlong":
            dropped_overlong += 1
        elif st == "no_assistant_tokens":
            dropped_no_targets += 1
        elif st == "ok":
            h = res["text_sha256"]
            if h in seen_hashes:
                dropped_duplicates += 1
                continue
            seen_hashes.add(h)
            kept_dedup.append(res)

    mismatch_frac = dropped_mismatch / max(1, len(filtered))
    if mismatch_frac > cfg.data.max_template_mismatch_fraction:
        raise DataContractError(
            f"Adapter {eff_adapter.name!r} template mismatch rate {mismatch_frac:.1%} "
            f"({dropped_mismatch}/{len(filtered)}) exceeds threshold {cfg.data.max_template_mismatch_fraction:.1%}"
        )

    if not kept_dedup:
        raise DataContractError(
            f"Adapter {eff_adapter.name!r} has 0 trainable examples after tokenization "
            f"(overlong={dropped_overlong}, template_mismatch={dropped_mismatch}, no_targets={dropped_no_targets})"
        )

    train_recs, val_recs = deterministic_task_split(
        kept_dedup,
        val_fraction=cfg.data.val_fraction,
        val_max_examples=cfg.data.val_max_examples,
        seed=cfg.run.seed,
    )

    save_parquet_split(train_recs, out_cache_dir / "train.parquet")
    save_parquet_split(val_recs, out_cache_dir / "val.parquet")

    lengths = sorted(int(r["length"]) for r in train_recs)
    train_sup_tokens = sum(int(r["supervised_tokens"]) for r in train_recs)
    train_total_tokens = sum(lengths)

    def _pct(p: float) -> int:
        if not lengths:
            return 0
        k = min(len(lengths) - 1, max(0, round(p * (len(lengths) - 1))))
        return lengths[k]

    summary = {
        "adapter": eff_adapter.name,
        "cache_key": cache_key,
        "roles": eff_adapter.roles,
        "raw_total": len(raw_records),
        "filtered_candidates": len(filtered),
        "missing_agent_field": missing_agent_count,
        "dropped_role_filter": dropped_role,
        "dropped_unresolved": dropped_unresolved,
        "dropped_overlong": dropped_overlong,
        "dropped_template_mismatch": dropped_mismatch,
        "dropped_no_assistant_tokens": dropped_no_targets,
        "dropped_duplicates": dropped_duplicates,
        "wire_format_warnings": wire_format_warnings,
        "train_examples": len(train_recs),
        "val_examples": len(val_recs),
        "train_total_tokens": train_total_tokens,
        "train_supervised_tokens": train_sup_tokens,
        "supervised_token_fraction": round(train_sup_tokens / max(1, train_total_tokens), 4),
        "length_histogram": {
            "min": lengths[0] if lengths else 0,
            "p50": _pct(0.50),
            "p90": _pct(0.90),
            "p99": _pct(0.99),
            "max": lengths[-1] if lengths else 0,
        },
    }
    (out_cache_dir / "cache_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class LengthGroupedLongestFirstSampler:
    """Deterministic sampler putting the longest sequences in the first optimizer steps (§7.5) and resumable (§7.8)."""

    def __init__(
        self,
        lengths: list[int],
        *,
        longest_first_samples: int = 0,
        mega_batch_size: int = 64,
        seed: int = 1234,
    ) -> None:
        self.lengths = [int(x) for x in lengths]
        self.n = len(self.lengths)
        self.longest_first_samples = max(0, min(self.n, int(longest_first_samples)))
        self.mega_batch_size = max(1, int(mega_batch_size))
        self.seed = int(seed)
        self.epoch: int = 0
        self.resume_offset: int = 0
        # Ledger-based resume: indices already trained on in `exclude_epoch` are skipped (order of the rest is kept).
        self.exclude: set[int] = set()
        self.exclude_epoch: int | None = None
        self._order: list[int] = []
        self._order_epoch: int = 0

    def set_exclusion(self, indices: Any, epoch: int) -> None:
        """Resume at `epoch` skipping `indices` (examples a previous session already consumed in that epoch)."""
        self.epoch = int(epoch)
        self.exclude = {int(i) for i in indices}
        self.exclude_epoch = int(epoch)
        self.resume_offset = 0

    def __len__(self) -> int:
        skipped = len(self.exclude) if self.epoch == self.exclude_epoch else 0
        return max(0, self.n - skipped - self.resume_offset)

    def consumed_indices(self, samples_consumed_in_epoch: int) -> list[int]:
        """Every index trained on in the epoch being iterated: earlier sessions' exclusions + the first N of this order."""
        prior = self.exclude if self._order_epoch == self.exclude_epoch else set()
        return sorted(prior | set(self._order[: max(0, int(samples_consumed_in_epoch))]))

    def build_epoch_order(self, epoch: int) -> list[int]:
        import random

        rng = random.Random(self.seed + int(epoch) * 100_003)
        all_indices = list(range(self.n))

        head: list[int] = []
        if epoch == 0 and self.longest_first_samples > 0:
            by_len_desc = sorted(all_indices, key=lambda i: (-self.lengths[i], i))
            head = by_len_desc[: self.longest_first_samples]
            head_set = set(head)
            rest = [i for i in all_indices if i not in head_set]
        else:
            rest = all_indices

        rng.shuffle(rest)
        buckets: list[list[int]] = []
        for start in range(0, len(rest), self.mega_batch_size):
            chunk = rest[start : start + self.mega_batch_size]
            chunk.sort(key=lambda i: -self.lengths[i])
            buckets.append(chunk)
        rng.shuffle(buckets)
        ordered = head + [i for b in buckets for i in b]
        return ordered

    def __iter__(self):
        order = self.build_epoch_order(self.epoch)
        if self.epoch == self.exclude_epoch and self.exclude:
            order = [i for i in order if i not in self.exclude]
        start = min(len(order), max(0, self.resume_offset))
        # Reset one-shot resume offset once consumed; advance epoch counter on next iteration
        self.resume_offset = 0
        current_epoch = self.epoch
        self._order, self._order_epoch = order, current_epoch
        self.epoch = current_epoch + 1
        return iter(order[start:])

    def state_dict(self, samples_consumed_in_epoch: int) -> dict[str, int]:
        return {
            "epoch": max(0, self.epoch - 1),
            "offset": int(samples_consumed_in_epoch) % max(1, self.n),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.resume_offset = int(state.get("offset", 0))


def check_tasks_file_optional(tasks_file: str | None) -> list[dict[str, Any]]:
    if not tasks_file:
        return []
    p = Path(tasks_file)
    if not p.is_file():
        return []
    return load_tasks(p)
