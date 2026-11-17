"""Offline quantitative evaluation & packaging script for EXP-016 (`swe-kit` Skill-First Agent).

Produces:
  1. Validated zero-adapter submission archives:
     - `runs/exp_016/submission_A0_pruned_6tools.zip` (Arm A0: 6 tools, 0 skills, 0.22 prompt)
     - `runs/exp_016/submission_A1_skill_first.zip`   (Arm A1: 6 tools + swe-kit + skill-first prompt)
     - `build/exp_016_submission.zip`                 (copy of Arm A1 ready for Kaggle upload)
  2. Full component benchmark on `MINED-389` (389 verified mined keep tasks) and held-out `DEV-30`:
     - Raw literal grep baseline vs Old `code_locator.py` vs New `swe-kit/locate.py`
     - Source Hit@1 / Hit@3 / Hit@5, Multi-file Full-Set Recall@5, `docs_src/` Hit@3,
       AST Symbol Span Hit Rate, Test Discovery Hit@3 / Hit@5, Latency (ms), Output Size (chars)
  3. Pre-submit gatekeeper benchmark across 7 patch archetypes × 30 real repo states (210 trials):
     - Old `patch_auditor.py` vs New `swe-kit/verify_and_audit.py`
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("PYTHONUTF8", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.package import (
    run_package_phase,
    validate_staged_submission,
)
from pipeline.multi_lora_sft.runctx import RunContext

UNIFIED_HEADER_RE = re.compile(
    r"^---\s+(?:a/)?([^\t\r\n]+).*?\r?\n\+\+\+\s+(?:b/)?([^\t\r\n]+)",
    re.MULTILINE,
)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _load_mod(name: str, path: Path) -> ModuleType:
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_patch_files_and_lines(patch_text: str) -> dict[str, list[int]]:
    """Extracts modified file paths and 1-indexed base-file line numbers from unified diffs (with or without `diff --git`)."""
    files_to_lines: dict[str, list[int]] = {}
    matches = list(UNIFIED_HEADER_RE.finditer(patch_text))
    for idx, m in enumerate(matches):
        a_raw = m.group(1).strip()
        b_raw = m.group(2).strip()
        rel = b_raw if b_raw != "/dev/null" else a_raw
        rel = rel.removeprefix("a/").removeprefix("b/")
        start_pos = m.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(patch_text)
        body = patch_text[start_pos:end_pos]
        touched: list[int] = []
        for hm in HUNK_RE.finditer(body):
            start_a = int(hm.group(1))
            count_a = int(hm.group(2)) if hm.group(2) is not None else 1
            if start_a > 0 and count_a > 0:
                touched.extend(range(start_a, start_a + count_a))
        files_to_lines[rel] = touched
    return files_to_lines


def _is_test_or_config(rel_p: str) -> bool:
    parts = rel_p.split("/")
    base = parts[-1]
    if base in {"conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"}:
        return True
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    return any(p in {"tests", "test", "testing"} for p in parts[:-1])


def gold_modified_ast_symbols(ws: Path, rel_p: str, base_lines: list[int]) -> set[str]:
    full_p = ws / rel_p
    if not full_p.is_file() or not base_lines:
        return set()
    try:
        tree = ast.parse(full_p.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError, OSError):
        return set()

    hit_syms: set[str] = set()

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = f"{prefix}.{child.name}" if prefix else child.name
                s_ln = child.lineno
                e_ln = getattr(child, "end_lineno", child.lineno) or child.lineno
                if any(s_ln <= ln <= e_ln for ln in base_lines):
                    hit_syms.add(qname)
                    hit_syms.add(child.name)
                if isinstance(child, ast.ClassDef):
                    _visit(child, qname)

    _visit(tree)
    return hit_syms


def raw_grep_rank_files(ws: Path, query: str, top_k: int = 5) -> list[str]:
    """Simulates the baseline agent's literal code-token grep over tracked source files."""
    code_toks: list[str] = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query):
        if (
            ("_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok)))
            and tok not in code_toks
        ):
            code_toks.append(tok)
        if len(code_toks) >= 10:
            break
    if not code_toks:
        for tok in re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", query):
            if tok not in code_toks:
                code_toks.append(tok)
            if len(code_toks) >= 8:
                break
    if not code_toks:
        return []

    counts: Counter[str] = Counter()
    for root, dirs, files in os.walk(ws):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", "docs", "tests", "test", "testing", ".venv", "build", "dist"}
            and not d.startswith(".")
        ]
        for f in files:
            if not f.endswith(".py") or f.startswith("test_") or f.endswith("_test.py"):
                continue
            fp = Path(root) / f
            rel = fp.relative_to(ws).as_posix()
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = sum(1 for t in code_toks if t in txt)
            if hits > 0:
                counts[rel] = hits
    return [rel for rel, _ in counts.most_common(top_k)]


def _eval_repo_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    repo_short: str = payload["repo_short"]
    bare_git_dir = Path(payload["bare_git_dir"])
    tasks: list[dict[str, Any]] = payload["tasks"]
    shard_id: int = payload["shard_id"]

    old_locator = _load_mod(
        "old_code_locator", REPO_ROOT / "data" / "Skills" / "code-locator" / "scripts" / "code_locator.py"
    )
    new_locator = _load_mod(
        "new_swe_locate", REPO_ROOT / "data" / "Skills" / "swe-kit" / "scripts" / "locate.py"
    )

    tmp_root = REPO_ROOT / "runs" / "exp_016" / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix=f"wt_{repo_short}_{shard_id}_", dir=tmp_root))
    wt_dir = tmp_base / "wt"
    results: list[dict[str, Any]] = []

    try:
        wt_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=wt_dir, check=True, capture_output=True)
        alt_file = wt_dir / ".git" / "objects" / "info" / "alternates"
        alt_file.parent.mkdir(parents=True, exist_ok=True)
        alt_file.write_bytes(((bare_git_dir / "objects").resolve().as_posix() + "\n").encode("utf-8"))

        for t in tasks:
            base_commit = t["base_commit"]
            co = subprocess.run(
                ["git", "-c", "core.longpaths=true", "checkout", "-f", "-q", base_commit],
                cwd=wt_dir,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if co.returncode != 0:
                continue

            patch_map = parse_patch_files_and_lines(t["patch"])
            gold_py_src = [
                p for p in patch_map
                if p.endswith(".py") and not _is_test_or_config(p)
            ]
            if not gold_py_src:
                continue

            test_map = parse_patch_files_and_lines(t.get("test_patch", ""))
            existing_gold_tests = [
                p for p in test_map
                if p.endswith(".py") and (wt_dir / p).is_file()
            ]

            gold_syms_by_file: dict[str, set[str]] = {
                gf: gold_modified_ast_symbols(wt_dir, gf, patch_map.get(gf, []))
                for gf in gold_py_src
            }
            has_any_gold_sym = any(bool(s) for s in gold_syms_by_file.values())

            query = t["problem_statement"]

            # 1. Raw literal grep baseline
            t0 = time.perf_counter()
            grep_top5 = raw_grep_rank_files(wt_dir, query, top_k=5)
            grep_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Old `code_locator.py`
            t0 = time.perf_counter()
            old_toks = old_locator.extract_tokens(query)
            old_cands = old_locator.find_candidate_files(wt_dir, old_toks, top_k=6) if old_toks else []
            old_src_ranked = [c["rel_file"] for c in old_cands if not c.get("is_test")][:5]
            old_tests = old_locator.find_relevant_tests(wt_dir, old_src_ranked[:3])
            old_ms = (time.perf_counter() - t0) * 1000.0
            old_json_chars = len(
                json.dumps(
                    {
                        "status": "ok",
                        "search_tokens": old_toks[:12],
                        "top_candidates": old_cands[:5],
                        "recommended_tests_to_run": old_tests,
                    },
                    indent=2,
                )
            )
            old_sym_hit = False
            for c in old_cands[:5]:
                rf = c.get("rel_file", "")
                if gold_syms_by_file.get(rf):
                    pred_names = {s.get("name", "") for s in c.get("symbols", [])}
                    if pred_names & gold_syms_by_file[rf]:
                        old_sym_hit = True
                        break

            # 3. New `swe-kit/locate.py`
            t0 = time.perf_counter()
            new_res = new_locator.rank_workspace(wt_dir, query=query, top_k=5)
            new_ms = (time.perf_counter() - t0) * 1000.0
            new_cands = new_res.get("top_candidates", [])
            new_src_ranked = [c["file"] for c in new_cands][:5]
            new_tests = new_res.get("recommended_tests", [])
            new_json_chars = len(json.dumps(new_res, indent=2))

            new_sym_hit = False
            for c in new_cands[:5]:
                rf = c.get("file", "")
                if gold_syms_by_file.get(rf):
                    pred_names = set()
                    for s in c.get("symbols", []):
                        pred_names.add(s["name"])
                        pred_names.add(s["name"].split(".")[-1])
                    if pred_names & gold_syms_by_file[rf]:
                        new_sym_hit = True
                        break

            gold_set = set(gold_py_src)
            gold_test_set = set(existing_gold_tests)
            touches_docs_src = any(g.startswith("docs_src/") for g in gold_py_src)

            results.append({
                "instance_id": t["instance_id"],
                "repo": repo_short,
                "tier": t.get("tier", "unknown"),
                "n_gold_src": len(gold_set),
                "touches_docs_src": touches_docs_src,
                "has_existing_gold_test": bool(gold_test_set),
                "has_gold_sym": has_any_gold_sym,
                # Raw grep
                "grep_hit1": bool(grep_top5[:1] and grep_top5[0] in gold_set),
                "grep_hit3": bool(set(grep_top5[:3]) & gold_set),
                "grep_hit5": bool(set(grep_top5[:5]) & gold_set),
                "grep_full5": len(set(grep_top5[:5]) & gold_set) / len(gold_set),
                "grep_ms": grep_ms,
                # Old code_locator
                "old_hit1": bool(old_src_ranked[:1] and old_src_ranked[0] in gold_set),
                "old_hit3": bool(set(old_src_ranked[:3]) & gold_set),
                "old_hit5": bool(set(old_src_ranked[:5]) & gold_set),
                "old_full5": len(set(old_src_ranked[:5]) & gold_set) / len(gold_set),
                "old_sym_hit": old_sym_hit,
                "old_test_hit3": bool(set(old_tests[:3]) & gold_test_set) if gold_test_set else None,
                "old_test_hit5": bool(set(old_tests[:5]) & gold_test_set) if gold_test_set else None,
                "old_ms": old_ms,
                "old_chars": old_json_chars,
                # New swe-kit locate.py
                "new_hit1": bool(new_src_ranked[:1] and new_src_ranked[0] in gold_set),
                "new_hit3": bool(set(new_src_ranked[:3]) & gold_set),
                "new_hit5": bool(set(new_src_ranked[:5]) & gold_set),
                "new_full5": len(set(new_src_ranked[:5]) & gold_set) / len(gold_set),
                "new_sym_hit": new_sym_hit,
                "new_test_hit3": bool(set(new_tests[:3]) & gold_test_set) if gold_test_set else None,
                "new_test_hit5": bool(set(new_tests[:5]) & gold_test_set) if gold_test_set else None,
                "new_ms": new_ms,
                "new_chars": new_json_chars,
            })
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    return results


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _pct(vals: list[bool]) -> float:
        return round(100.0 * sum(1 for v in vals if v) / len(vals), 1) if vals else 0.0

    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def _p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(0.95 * len(s)))], 1)

    docs_rows = [r for r in rows if r["touches_docs_src"]]
    multi_rows = [r for r in rows if r["n_gold_src"] >= 2]
    sym_rows = [r for r in rows if r["has_gold_sym"]]
    test_rows = [r for r in rows if r["has_existing_gold_test"]]

    return {
        "n_tasks": n,
        "n_docs_src": len(docs_rows),
        "n_multi_file": len(multi_rows),
        "n_with_existing_test": len(test_rows),
        "raw_grep": {
            "hit1_pct": _pct([r["grep_hit1"] for r in rows]),
            "hit3_pct": _pct([r["grep_hit3"] for r in rows]),
            "hit5_pct": _pct([r["grep_hit5"] for r in rows]),
            "multi_full_recall5_pct": round(100.0 * sum(r["grep_full5"] for r in multi_rows) / len(multi_rows), 1)
            if multi_rows else 0.0,
            "docs_src_hit3_pct": _pct([r["grep_hit3"] for r in docs_rows]),
            "mean_ms": _avg([r["grep_ms"] for r in rows]),
        },
        "old_code_locator": {
            "hit1_pct": _pct([r["old_hit1"] for r in rows]),
            "hit3_pct": _pct([r["old_hit3"] for r in rows]),
            "hit5_pct": _pct([r["old_hit5"] for r in rows]),
            "multi_full_recall5_pct": round(100.0 * sum(r["old_full5"] for r in multi_rows) / len(multi_rows), 1)
            if multi_rows else 0.0,
            "docs_src_hit3_pct": _pct([r["old_hit3"] for r in docs_rows]),
            "ast_symbol_hit_pct": _pct([r["old_sym_hit"] for r in sym_rows]),
            "test_hit3_pct": _pct([bool(r["old_test_hit3"]) for r in test_rows]),
            "test_hit5_pct": _pct([bool(r["old_test_hit5"]) for r in test_rows]),
            "mean_ms": _avg([r["old_ms"] for r in rows]),
            "p95_ms": _p95([r["old_ms"] for r in rows]),
            "mean_chars": round(statistics.mean(r["old_chars"] for r in rows), 0),
        },
        "exp016_swe_kit": {
            "hit1_pct": _pct([r["new_hit1"] for r in rows]),
            "hit3_pct": _pct([r["new_hit3"] for r in rows]),
            "hit5_pct": _pct([r["new_hit5"] for r in rows]),
            "multi_full_recall5_pct": round(100.0 * sum(r["new_full5"] for r in multi_rows) / len(multi_rows), 1)
            if multi_rows else 0.0,
            "docs_src_hit3_pct": _pct([r["new_hit3"] for r in docs_rows]),
            "ast_symbol_hit_pct": _pct([r["new_sym_hit"] for r in sym_rows]),
            "test_hit3_pct": _pct([bool(r["new_test_hit3"]) for r in test_rows]),
            "test_hit5_pct": _pct([bool(r["new_test_hit5"]) for r in test_rows]),
            "mean_ms": _avg([r["new_ms"] for r in rows]),
            "p95_ms": _p95([r["new_ms"] for r in rows]),
            "mean_chars": round(statistics.mean(r["new_chars"] for r in rows), 0),
        },
    }


def benchmark_auditors_on_real_repos(mined_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Benchmarks old `patch_auditor.py` vs new `swe-kit/verify_and_audit.py` across 30 real repo states × 7 scenarios."""
    old_auditor = _load_mod(
        "old_patch_auditor", REPO_ROOT / "data" / "Skills" / "patch-auditor" / "scripts" / "patch_auditor.py"
    )
    new_auditor = _load_mod(
        "new_verify_and_audit", REPO_ROOT / "data" / "Skills" / "swe-kit" / "scripts" / "verify_and_audit.py"
    )

    # Pick up to 8 mined tasks per repo deterministically (total 30 tasks)
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in mined_tasks:
        by_repo[t["repo_short"]].append(t)

    sampled: list[dict[str, Any]] = []
    for r_name in ("fastapi", "rich", "httpx", "requests"):
        sampled.extend(by_repo[r_name][:8])
    sampled = sampled[:30]

    scenarios = [
        "S0_valid_gold_patch",
        "S1_empty_tree",
        "S2_stray_root_repro_call1",
        "S3_stray_root_repro_call2",
        "S4_tests_only_tamper",
        "S5_config_only_tamper",
        "S6_syntax_error",
    ]
    old_correct: Counter[str] = Counter()
    new_correct: Counter[str] = Counter()
    totals: Counter[str] = Counter()

    tmp_root = REPO_ROOT / "runs" / "exp_016" / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix="audit_", dir=tmp_root))
    try:
        for repo_short in ("fastapi", "rich", "httpx", "requests"):
            repo_tasks = [t for t in sampled if t["repo_short"] == repo_short]
            if not repo_tasks:
                continue
            bare_git = REPO_ROOT / "DataMining" / "raw_data" / "upstream" / f"{repo_short}.git"
            wt = tmp_base / f"wt_{repo_short}"
            wt.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q"], cwd=wt, check=True, capture_output=True)
            subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=wt, check=True, capture_output=True)
            alt_file = wt / ".git" / "objects" / "info" / "alternates"
            alt_file.parent.mkdir(parents=True, exist_ok=True)
            alt_file.write_bytes(((bare_git / "objects").resolve().as_posix() + "\n").encode("utf-8"))

            for t in repo_tasks:
                base_c = t["base_commit"]

                def _reset(wt_path: Path = wt, commit_sha: str = base_c) -> None:
                    subprocess.run(
                        ["git", "-c", "core.longpaths=true", "checkout", "-f", "-q", commit_sha],
                        cwd=wt_path,
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(["git", "clean", "-fdq"], cwd=wt_path, check=True, capture_output=True)

                patch_files = [
                    p for p in parse_patch_files_and_lines(t["patch"])
                    if p.endswith(".py") and not _is_test_or_config(p)
                ]
                if not patch_files:
                    continue
                primary_src = patch_files[0]

                # S0: Valid gold source patch (expected: ready_to_submit == True)
                _reset()
                raw_p = t["patch"] if t["patch"].endswith("\n") else t["patch"] + "\n"
                ap = subprocess.run(
                    ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn", "-"],
                    input=raw_p,
                    cwd=wt,
                    text=True,
                    encoding="utf-8",
                    check=False,
                    capture_output=True,
                )
                if ap.returncode == 0:
                    totals["S0_valid_gold_patch"] += 1
                    if old_auditor.audit_workspace(wt)["ready_to_submit"] is True:
                        old_correct["S0_valid_gold_patch"] += 1
                    if new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"] is True:
                        new_correct["S0_valid_gold_patch"] += 1

                # S1: Empty unmodified tree (expected: ready_to_submit == False)
                _reset()
                totals["S1_empty_tree"] += 1
                if old_auditor.audit_workspace(wt)["ready_to_submit"] is False:
                    old_correct["S1_empty_tree"] += 1
                if new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"] is False:
                    new_correct["S1_empty_tree"] += 1

                # S2 & S3: Valid source edit + stray `/workspace/repro.py` on call 1 and call 2 (expected: False)
                _reset()
                if (wt / primary_src).is_file():
                    orig_txt = (wt / primary_src).read_text(encoding="utf-8", errors="ignore")
                    (wt / primary_src).write_text(orig_txt + "\n# surgical fix comment\n", encoding="utf-8")
                    (wt / "repro.py").write_text("print('scratch repro')\n", encoding="utf-8")
                    totals["S2_stray_root_repro_call1"] += 1
                    totals["S3_stray_root_repro_call2"] += 1
                    r_old_1 = old_auditor.audit_workspace(wt)["ready_to_submit"]
                    r_old_2 = old_auditor.audit_workspace(wt)["ready_to_submit"]
                    if r_old_1 is False:
                        old_correct["S2_stray_root_repro_call1"] += 1
                    if r_old_2 is False:
                        old_correct["S3_stray_root_repro_call2"] += 1

                    # Reset index intent-to-add before testing new_auditor on clean untracked state, then call twice
                    _reset()
                    (wt / primary_src).write_text(orig_txt + "\n# surgical fix comment\n", encoding="utf-8")
                    (wt / "repro.py").write_text("print('scratch repro')\n", encoding="utf-8")
                    r_new_1 = new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"]
                    r_new_2 = new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"]
                    if r_new_1 is False:
                        new_correct["S2_stray_root_repro_call1"] += 1
                    if r_new_2 is False:
                        new_correct["S3_stray_root_repro_call2"] += 1

                # S4: Tests-only tamper (expected: ready_to_submit == False because grader strips tests/)
                _reset()
                test_candidates = sorted((wt / "tests").rglob("test_*.py")) if (wt / "tests").is_dir() else []
                if test_candidates:
                    tf = test_candidates[0]
                    tf.write_text(tf.read_text(encoding="utf-8", errors="ignore") + "\n# tampered test\n", encoding="utf-8")
                    totals["S4_tests_only_tamper"] += 1
                    if old_auditor.audit_workspace(wt)["ready_to_submit"] is False:
                        old_correct["S4_tests_only_tamper"] += 1
                    if new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"] is False:
                        new_correct["S4_tests_only_tamper"] += 1

                # S5: Protected runner config tamper (`pyproject.toml` / `setup.cfg`) (expected: False)
                _reset()
                cfg_file = wt / "pyproject.toml" if (wt / "pyproject.toml").is_file() else wt / "setup.cfg"
                if cfg_file.is_file():
                    cfg_file.write_text(cfg_file.read_text(encoding="utf-8", errors="ignore") + "\n# tampered\n", encoding="utf-8")
                    totals["S5_config_only_tamper"] += 1
                    if old_auditor.audit_workspace(wt)["ready_to_submit"] is False:
                        old_correct["S5_config_only_tamper"] += 1
                    if new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"] is False:
                        new_correct["S5_config_only_tamper"] += 1

                # S6: SyntaxError in source file (expected: False)
                _reset()
                if (wt / primary_src).is_file():
                    (wt / primary_src).write_text("def broken_syntax(\n    return 1\n", encoding="utf-8")
                    totals["S6_syntax_error"] += 1
                    if old_auditor.audit_workspace(wt)["ready_to_submit"] is False:
                        old_correct["S6_syntax_error"] += 1
                    if new_auditor.verify_and_audit(wt, no_tests=True)["ready_to_submit"] is False:
                        new_correct["S6_syntax_error"] += 1
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    summary: dict[str, Any] = {}
    total_trials = sum(totals.values())
    for sc in scenarios:
        n = totals[sc]
        summary[sc] = {
            "trials": n,
            "old_patch_auditor_accuracy_pct": round(100.0 * old_correct[sc] / n, 1) if n else 0.0,
            "exp016_verify_and_audit_accuracy_pct": round(100.0 * new_correct[sc] / n, 1) if n else 0.0,
        }
    summary["overall"] = {
        "total_trials": total_trials,
        "old_patch_auditor_accuracy_pct": round(100.0 * sum(old_correct.values()) / total_trials, 1),
        "exp016_verify_and_audit_accuracy_pct": round(100.0 * sum(new_correct.values()) / total_trials, 1),
    }
    return summary


def package_exp016_submissions() -> dict[str, Any]:
    out_root = REPO_ROOT / "runs" / "exp_016"
    out_root.mkdir(parents=True, exist_ok=True)
    tpl_dir = REPO_ROOT / "pipeline" / "submission_template"

    # Build temporary A0 template (6 pruned tools, 0 skills, proven 0.22 prompt without graph-tool ban line)
    a0_tpl = out_root / "_tpl_a0"
    if a0_tpl.exists():
        shutil.rmtree(a0_tpl)
    shutil.copytree(tpl_dir, a0_tpl)
    with zipfile.ZipFile(REPO_ROOT / "kaggle" / "submission.zip") as z_ctrl:
        ctrl_prompt = z_ctrl.read("prompts/system.md").decode("utf-8")
    # Remove the obsolete line banning the 3 graph tools since they are pruned from agent.yaml
    a0_prompt = "\n".join(
        ln for ln in ctrl_prompt.splitlines() if "search_similar_code" not in ln
    ) + "\n"
    (a0_tpl / "prompts" / "system.md").write_text(a0_prompt, encoding="utf-8")

    # Package Arm A0
    cfg_a0 = load_config(
        set_overrides=[
            f"run.output_root={out_root.as_posix()}",
            "run.run_id=arm_a0_pruned_6tools",
            "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
            f"submission.template_dir={a0_tpl.as_posix()}",
            "submission.write_generation_config=false",
            "submission.write_eval_config=false",
            "submission.skills=[]",
            f"submission.working_zip_path={(out_root / 'submission_A0_pruned_6tools.zip').as_posix()}",
        ]
    )
    ctx_a0 = RunContext.create(cfg_a0, t0=time.time())
    cfg_a0.save_yaml(ctx_a0.resolved_config_path)
    st_a0 = run_package_phase(ctx_a0, cfg=cfg_a0)
    val_a0 = validate_staged_submission(ctx_a0.submission_dir, expected_served_model="gemma-4-31b-it-qat-w4a16-ct")
    shutil.rmtree(a0_tpl, ignore_errors=True)

    # Package Arm A1 (EXP-016 Skill-First Agent)
    cfg_a1 = load_config(
        set_overrides=[
            f"run.output_root={out_root.as_posix()}",
            "run.run_id=arm_a1_skill_first",
            "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
            f"submission.template_dir={tpl_dir.as_posix()}",
            "submission.write_generation_config=false",
            "submission.write_eval_config=false",
            "submission.skills=[swe-kit]",
            f"submission.working_zip_path={(out_root / 'submission_A1_skill_first.zip').as_posix()}",
        ]
    )
    ctx_a1 = RunContext.create(cfg_a1, t0=time.time())
    cfg_a1.save_yaml(ctx_a1.resolved_config_path)
    st_a1 = run_package_phase(ctx_a1, cfg=cfg_a1)
    val_a1 = validate_staged_submission(ctx_a1.submission_dir, expected_served_model="gemma-4-31b-it-qat-w4a16-ct")

    build_copy = REPO_ROOT / "build" / "exp_016_submission.zip"
    build_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_root / "submission_A1_skill_first.zip", build_copy)

    return {
        "CTRL_31B": {
            "zip_path": "kaggle/submission.zip",
            "zip_bytes": (REPO_ROOT / "kaggle" / "submission.zip").stat().st_size,
            "declared_tools": 9,
            "graph_tools_declared": 3,
            "skills_declared": 0,
            "system_prompt_chars": len(ctrl_prompt),
        },
        "Arm_A0_pruned_6tools": {
            "zip_path": "runs/exp_016/submission_A0_pruned_6tools.zip",
            "zip_bytes": st_a0["outputs"]["submission_zip_bytes"],
            "unpacked_bytes": val_a0["unpacked_size_bytes"],
            "file_count": val_a0["file_count"],
            "declared_tools": 6,
            "graph_tools_declared": 0,
            "skills_declared": 0,
            "system_prompt_chars": len(a0_prompt),
            "sha256": st_a0["outputs"]["submission_zip_sha256"],
        },
        "Arm_A1_skill_first": {
            "zip_path": "runs/exp_016/submission_A1_skill_first.zip",
            "build_zip_path": "build/exp_016_submission.zip",
            "zip_bytes": st_a1["outputs"]["submission_zip_bytes"],
            "unpacked_bytes": val_a1["unpacked_size_bytes"],
            "file_count": val_a1["file_count"],
            "declared_tools": 6,
            "graph_tools_declared": 0,
            "skills_declared": 1,
            "skill_scripts": ["locate.py", "read_symbol.py", "verify_and_audit.py"],
            "system_prompt_chars": len((tpl_dir / "prompts" / "system.md").read_text(encoding="utf-8")),
            "sha256": st_a1["outputs"]["submission_zip_sha256"],
        },
    }


def load_mined_389_tasks() -> list[dict[str, Any]]:
    mined_root = REPO_ROOT / "DataMining" / "raw_data" / "mined"
    tasks: list[dict[str, Any]] = []
    for sub in ("prewindow", "postwindow", "httpx", "inwindow40"):
        q_map: dict[str, dict[str, Any]] = {}
        with (mined_root / sub / "quality_sheet.jsonl").open(encoding="utf-8", newline="\n") as qf:
            for ln in qf:
                if ln.strip():
                    q = json.loads(ln)
                    if q.get("verdict") == "keep":
                        q_map[q["instance_id"]] = q
        with (mined_root / sub / "tasks.jsonl").open(encoding="utf-8", newline="\n") as tf:
            for ln in tf:
                if not ln.strip():
                    continue
                t = json.loads(ln)
                iid = t["instance_id"]
                if iid in q_map:
                    repo_short = t["repo"].split("/")[-1].lower()
                    t["repo_short"] = repo_short
                    t["tier"] = q_map[iid].get("tier", "unknown")
                    tasks.append(t)
    return tasks


def load_dev30_tasks() -> list[dict[str, Any]]:
    dev_meta = json.loads(
        (REPO_ROOT / "experiments" / "exp_016_skill_first_agent" / "dev_30_ids.json").read_text(encoding="utf-8")
    )
    tier_by_id = {s["instance_id"]: s["tier"] for s in dev_meta["strata"]}
    wanted = set(dev_meta["task_ids"])
    tasks_path = REPO_ROOT / "build" / "gemma-4-developer-agent" / "tasks.jsonl"
    out: list[dict[str, Any]] = []
    with tasks_path.open(encoding="utf-8", newline="\n") as tf:
        for ln in tf:
            if not ln.strip():
                continue
            t = json.loads(ln)
            iid = t["instance_id"]
            if iid in wanted:
                t["repo_short"] = t["repo"].split("/")[-1].lower()
                t["tier"] = tier_by_id[iid]
                out.append(t)
    return out


def run_dataset_eval(tasks: list[dict[str, Any]], max_workers: int = 12) -> dict[str, Any]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        by_repo[t["repo_short"]].append(t)

    shards: list[dict[str, Any]] = []
    shard_id = 0
    for repo_short, r_tasks in sorted(by_repo.items()):
        bare_git = REPO_ROOT / "DataMining" / "raw_data" / "upstream" / f"{repo_short}.git"
        chunk_sz = 28 if len(r_tasks) > 35 else max(1, len(r_tasks))
        for i in range(0, len(r_tasks), chunk_sz):
            shards.append({
                "shard_id": shard_id,
                "repo_short": repo_short,
                "bare_git_dir": str(bare_git),
                "tasks": r_tasks[i : i + chunk_sz],
            })
            shard_id += 1

    workers = min(max_workers, os.cpu_count() or 8, len(shards))
    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for res_list in pool.map(_eval_repo_shard, shards):
            all_rows.extend(res_list)

    by_repo_summary = {
        r_name: summarize_rows([r for r in all_rows if r["repo"] == r_name])
        for r_name in sorted({r["repo"] for r in all_rows})
    }
    by_tier_summary = {
        tier: summarize_rows([r for r in all_rows if r["tier"] == tier])
        for tier in sorted({r["tier"] for r in all_rows})
    }
    return {
        "overall": summarize_rows(all_rows),
        "by_repo": by_repo_summary,
        "by_tier": by_tier_summary,
    }


def main() -> int:
    t0 = time.perf_counter()
    print("[1/4] Packaging and validating EXP-016 submission archives (Arm A0 & Arm A1)...")
    pkg_report = package_exp016_submissions()

    print("[2/4] Evaluating localization, AST symbol spans & test discovery on MINED-389...")
    mined_tasks = load_mined_389_tasks()
    mined_report = run_dataset_eval(mined_tasks, max_workers=12)

    print("[3/4] Evaluating localization, AST symbol spans & test discovery on held-out DEV-30...")
    dev30_tasks = load_dev30_tasks()
    dev30_report = run_dataset_eval(dev30_tasks, max_workers=8)

    print("[4/4] Benchmarking pre-submit patch auditor gatekeeper across 30 real repo states × 7 scenarios...")
    audit_report = benchmark_auditors_on_real_repos(mined_tasks)

    elapsed = round(time.perf_counter() - t0, 2)
    full_report = {
        "experiment": "EXP-016",
        "elapsed_seconds": elapsed,
        "packaging": pkg_report,
        "mined_389": mined_report,
        "dev_30": dev30_report,
        "patch_auditor_gatekeeper": audit_report,
    }

    out_json = REPO_ROOT / "runs" / "exp_016" / "component_eval_results.json"
    out_json.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved complete quantitative report to {out_json} in {elapsed}s.\n")
    print(json.dumps(full_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
