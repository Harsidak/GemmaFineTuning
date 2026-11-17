#!/usr/bin/env python3
"""BM25 + AST Symbol & Test Locator for SWE-bench repositories (stdlib only).

Ranks package source files (including docs_src/) and maps them to relevant unit tests
using Inverse Document Frequency (IDF), AST symbol definitions, and import scanning.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "site-packages",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    "node_modules",
    "htmlcov",
    "docs",  # markdown/translations; executable tutorials live in docs_src/
    "benchmarks",
}

GENERIC_STEMS = {
    "__init__",
    "__main__",
    "_compat",
    "base",
    "common",
    "compat",
    "config",
    "constants",
    "core",
    "errors",
    "exceptions",
    "helpers",
    "main",
    "misc",
    "models",
    "structures",
    "types",
    "typing",
    "utils",
}

STOPWORDS = {
    "a", "about", "above", "actual", "add", "added", "adding", "adds", "after", "again",
    "all", "allow", "allows", "also", "always", "am", "an", "and", "any", "are", "aren",
    "as", "at", "be", "because", "been", "before", "behavior", "behaviour", "being",
    "below", "between", "both", "branch", "bug", "build", "but", "by", "can", "cannot",
    "case", "cases", "change", "changed", "changes", "check", "checklist", "chore",
    "clean", "close", "closes", "code", "commit", "context", "could", "current",
    "currently", "default", "defaults", "define", "defined", "description", "details",
    "did", "didn", "do", "docs", "documentation", "does", "doesn", "doing", "don",
    "done", "down", "du", "during", "each", "eg", "either", "else", "enable", "enabled",
    "ensure", "error", "errors", "etc", "even", "every", "example", "examples",
    "expected", "fail", "failed", "failing", "fails", "failure", "false", "feat",
    "feature", "file", "files", "fix", "fixed", "fixes", "fixing", "following", "for",
    "from", "function", "functions", "further", "get", "gets", "getting", "gh", "git",
    "github", "give", "given", "got", "had", "handle", "handled", "handles", "handling",
    "has", "have", "having", "he", "help", "here", "how", "however", "http", "https",
    "id", "ie", "if", "ignore", "implement", "implemented", "import", "in", "include",
    "included", "includes", "instead", "into", "is", "isn", "issue", "issues", "it",
    "its", "itself", "just", "keep", "let", "like", "line", "lines", "link", "list",
    "made", "make", "makes", "making", "may", "me", "merge", "merged", "method",
    "methods", "might", "more", "most", "move", "must", "my", "name", "need", "needed",
    "needs", "new", "next", "no", "none", "nor", "not", "note", "notes", "now", "null",
    "number", "object", "of", "off", "ok", "old", "on", "once", "one", "only", "open",
    "option", "optional", "or", "other", "our", "out", "output", "over", "own",
    "package", "param", "parameter", "parameters", "pass", "passed", "passes", "patch",
    "path", "perf", "please", "possible", "pr", "prevent", "problem", "proper",
    "properly", "provide", "provided", "pull", "py", "python", "raise", "raised",
    "raises", "read", "readme", "refactor", "ref", "related", "release", "remove",
    "removed", "replace", "replaced", "repo", "repository", "reproduce", "request",
    "requests_this", "require", "required", "resolve", "resolved", "resolves", "result",
    "results", "return", "returned", "returns", "right", "run", "running", "runs",
    "same", "see", "seem", "seems", "self", "set", "sets", "setting", "settings",
    "should", "shouldn", "show", "shows", "simple", "since", "small", "so", "some",
    "something", "specific", "still", "str", "string", "style", "sub", "submit", "such",
    "summary", "support", "supported", "supports", "sure", "take", "takes", "test",
    "tested", "testing", "tests", "than", "thank", "thanks", "that", "the", "their",
    "them", "then", "there", "these", "they", "thing", "things", "this", "those",
    "through", "time", "title", "to", "todo", "too", "true", "try", "trying", "type",
    "types", "under", "until", "up", "update", "updated", "updates", "upgrade", "url",
    "usage", "use", "used", "user", "users", "uses", "using", "value", "values", "var",
    "version", "versions", "very", "via", "want", "was", "wasn", "way", "we", "well",
    "were", "weren", "what", "when", "where", "whether", "which", "while", "who", "why",
    "will", "with", "within", "without", "work", "working", "works", "would", "wouldn",
    "write", "wrong", "yet", "you", "your",
}

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PATH_ANCHOR_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py\b|\b[A-Za-z_][A-Za-z0-9_]*\.py\b")
DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")


def _normalize_argv(raw_argv: list[str]) -> list[str]:
    if not raw_argv:
        env_args = os.environ.get("SKILL_ARGS") or os.environ.get("ARGS") or ""
        if env_args.strip():
            raw_argv = [env_args.strip()]
        else:
            return []

    if len(raw_argv) == 1:
        single = raw_argv[0].strip()
        if single.startswith("{") and single.endswith("}"):
            try:
                obj = json.loads(single)
                if isinstance(obj, dict):
                    expanded: list[str] = []
                    if "query" in obj:
                        expanded.extend(["--query", str(obj["query"])])
                    if "workspace" in obj:
                        expanded.extend(["--workspace", str(obj["workspace"])])
                    if "top_k" in obj or "top-k" in obj:
                        expanded.extend(["--top-k", str(obj.get("top_k", obj.get("top-k")))])
                    return expanded
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if single.startswith("-") and " " in single:
            try:
                return shlex.split(single)
            except ValueError:
                pass
    return raw_argv


def _strip_pr_boilerplate(text: str) -> str:
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Remove AI disclaimer blocks common in newer FastAPI PRs
    text = re.sub(r"#{1,4}\s*AI Disclaimer.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove markdown checklist items
    text = re.sub(r"^\s*[-*]\s*\[[ xX]\]\s*.*$", " ", text, flags=re.MULTILINE)
    # Remove URLs and Fixes/Closes #NNN
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b(?:Fixes|Closes|Resolves|Part of|Refs?)\s+#?\d+\b", " ", text, flags=re.IGNORECASE)
    return text


def _is_code_shaped(tok: str) -> bool:
    if "_" in tok:
        return True
    if any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok):
        return True
    return tok.endswith(("Error", "Exception", "Warning", "Middleware", "Router", "Config", "Model", "Handler"))


def extract_query_features(query: str) -> dict[str, Any]:
    cleaned = _strip_pr_boilerplate(query)
    wants_docs_src = bool(re.search(r"\b(?:docs_src|tutorial\d*|docs/)\b", query, flags=re.IGNORECASE))

    path_anchors = [m.group(0).lstrip("./") for m in PATH_ANCHOR_RE.finditer(cleaned)]
    backtick_spans = BACKTICK_RE.findall(cleaned)
    backtick_tokens: set[str] = set()
    for span in backtick_spans:
        for t in WORD_RE.findall(span):
            backtick_tokens.add(t)

    dotted_parts: set[str] = set()
    for dm in DOTTED_RE.finditer(cleaned):
        full = dm.group(0)
        # Skip version-like or domain-like strings
        parts = full.split(".")
        if all(p.isidentifier() for p in parts):
            for p in parts:
                if len(p) >= 2 and p.lower() not in STOPWORDS:
                    dotted_parts.add(p)

    weights: dict[str, float] = {}

    def _add_tok(tok: str, base_w: float) -> None:
        tl = tok.lower()
        if tl in STOPWORDS:
            return
        if len(tok) < 2 or (len(tok) == 2 and not (_is_code_shaped(tok) or tok in backtick_tokens or tok.isupper())):
            return
        if tok.isdigit():
            return
        w = base_w
        if _is_code_shaped(tok):
            w *= 2.2
        if tok in backtick_tokens:
            w *= 1.6
        if tok in dotted_parts:
            w *= 1.5
        if w > weights.get(tok, 0.0):
            weights[tok] = w

    # Title / first non-empty line gets higher weight because PR titles summarize the exact target
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    title_text = lines[0] if lines else cleaned[:200]
    for tok in WORD_RE.findall(title_text):
        _add_tok(tok, 1.6)

    for tok in WORD_RE.findall(cleaned):
        _add_tok(tok, 1.0)

    # Expand camelCase and snake_case sub-parts at a lower weight for recall
    for tok, parent_w in list(weights.items()):
        subparts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", tok.replace("_", " "))
        if len(subparts) > 1:
            for sp in subparts:
                if len(sp) >= 3 and sp.lower() not in STOPWORDS and not sp.isdigit():
                    sub_w = max(0.65, parent_w * 0.35)
                    if sub_w > weights.get(sp, 0.0):
                        weights[sp] = sub_w

    # Keep top 36 highest-weight tokens (deduplicating case-insensitively while preferring exact case)
    by_lower: dict[str, tuple[str, float]] = {}
    for tok, w in sorted(weights.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
        tl = tok.lower()
        if tl not in by_lower or w > by_lower[tl][1]:
            by_lower[tl] = (tok, w)

    ranked_tokens = sorted(by_lower.values(), key=lambda kv: (-kv[1], -len(kv[0])))[:36]
    return {
        "tokens": dict(ranked_tokens),
        "path_anchors": list(dict.fromkeys(path_anchors)),
        "wants_docs_src": wants_docs_src,
    }


def _is_test_relpath(rel_posix: str, filename: str) -> bool:
    parts = rel_posix.split("/")
    if any(p in {"tests", "test", "testing"} for p in parts[:-1]):
        return True
    return filename.startswith("test_") or filename.endswith("_test.py") or filename == "conftest.py"


def _collect_repo_py_files(workspace: Path) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    source_files: list[tuple[Path, str]] = []
    test_files: list[tuple[Path, str]] = []
    skip_root_files = {"setup.py", "noxfile.py", "fabfile.py", "conftest.py"}

    for root, dirs, files in os.walk(workspace):
        rel_root = Path(root).relative_to(workspace).as_posix()
        dirs[:] = sorted(
            d for d in dirs
            if d not in IGNORE_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
        )
        for fname in sorted(files):
            if not fname.endswith(".py") or fname.startswith("."):
                continue
            full_p = Path(root) / fname
            rel_p = full_p.relative_to(workspace).as_posix()
            if rel_root == "." and fname in skip_root_files:
                continue
            if rel_p.startswith(("scripts/", "examples/")):
                continue
            if _is_test_relpath(rel_p, fname):
                test_files.append((full_p, rel_p))
            else:
                source_files.append((full_p, rel_p))
    return source_files, test_files


def _extract_ast_symbols_and_enclosing(
    content: str,
    query_tokens: dict[str, float],
    matched_line_nums: list[int],
) -> tuple[float, list[dict[str, Any]]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return 0.0, []

    all_defs: list[dict[str, Any]] = []

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = f"{prefix}.{child.name}" if prefix else child.name
                kind = "class" if isinstance(child, ast.ClassDef) else "def"
                end_ln = getattr(child, "end_lineno", child.lineno) or child.lineno
                all_defs.append({
                    "name": qname,
                    "short_name": child.name,
                    "kind": kind,
                    "start_line": child.lineno,
                    "end_line": end_ln,
                })
                if isinstance(child, ast.ClassDef):
                    _visit(child, qname)

    _visit(tree)

    q_lower = {k.lower(): (k, w) for k, w in query_tokens.items()}
    ast_bonus = 0.0
    scored_syms: list[tuple[float, dict[str, Any]]] = []

    for sym in all_defs:
        sname = sym["short_name"]
        sl = sname.lower()
        sym_score = 0.0
        if sl in q_lower:
            orig_tok, w = q_lower[sl]
            sym_score += (14.0 if orig_tok == sname else 10.0) * w
        else:
            for tl, (_, w) in q_lower.items():
                if len(tl) >= 4 and tl in sl:
                    sym_score += 3.0 * w

        # Also credit symbols enclosing matched lines
        encloses = any(
            sym["start_line"] <= ln <= sym["end_line"] and sym["kind"] == "def"
            for ln in matched_line_nums
        )
        if encloses:
            sym_score += 2.5

        if sym_score > 0:
            ast_bonus += min(sym_score, 30.0)
            scored_syms.append((sym_score, {
                "name": sym["name"],
                "kind": sym["kind"],
                "lines": [sym["start_line"], sym["end_line"]],
            }))

    scored_syms.sort(key=lambda x: (-x[0], x[1]["lines"][0]))
    seen: set[str] = set()
    out_syms: list[dict[str, Any]] = []
    for _, s in scored_syms:
        if s["name"] not in seen:
            seen.add(s["name"])
            out_syms.append(s)
            if len(out_syms) >= 4:
                break
    return min(ast_bonus, 60.0), out_syms


def rank_workspace(
    workspace: Path,
    query: str,
    top_k: int = 6,
) -> dict[str, Any]:
    feats = extract_query_features(query)
    q_tokens: dict[str, float] = feats["tokens"]
    path_anchors: list[str] = feats["path_anchors"]
    wants_docs_src: bool = feats["wants_docs_src"]

    source_files, test_files = _collect_repo_py_files(workspace)
    if not source_files:
        return {"status": "error", "message": f"No Python source files found in {workspace}"}

    if not q_tokens and not path_anchors:
        return {"status": "error", "message": "No meaningful identifiers or keywords found in query"}

    q_lower_to_orig = {t.lower(): (t, w) for t, w in q_tokens.items()}
    q_lowers = set(q_lower_to_orig)

    # Single-pass C-speed tokenization over source files
    doc_records: list[dict[str, Any]] = []
    df_counter: Counter[str] = Counter()
    total_len = 0

    for full_p, rel_p in source_files:
        try:
            content = full_p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        words = WORD_RE.findall(content)
        doc_len = max(1, len(words))
        total_len += doc_len
        exact_counts = Counter(w for w in words if w.lower() in q_lowers)
        lower_counts: Counter[str] = Counter()
        for w, c in exact_counts.items():
            lower_counts[w.lower()] += c

        for tl in lower_counts:
            df_counter[tl] += 1

        doc_records.append({
            "path": full_p,
            "rel": rel_p,
            "content": content,
            "doc_len": doc_len,
            "exact_counts": exact_counts,
            "lower_counts": lower_counts,
        })

    n_docs = max(1, len(doc_records))
    avgdl = max(1.0, total_len / n_docs)
    k1 = 1.5
    b = 0.35

    idf: dict[str, float] = {}
    for tl in q_lowers:
        df = df_counter.get(tl, 0)
        idf[tl] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    prelim: list[dict[str, Any]] = []
    for rec in doc_records:
        rel_p = rec["rel"]
        rel_lower = rel_p.lower()
        stem_lower = Path(rel_p).stem.lower()
        path_parts_lower = {p.lower() for p in Path(rel_p).with_suffix("").parts}

        score = 0.0
        matched_toks: list[str] = []

        # 1. Explicit path anchor match
        for pa in path_anchors:
            pa_l = pa.lower()
            if rel_lower == pa_l or rel_lower.endswith("/" + pa_l):
                score += 65.0
            elif "/" not in pa_l and Path(rel_p).name.lower() == pa_l:
                score += 30.0

        # 2. BM25 lexical + identifier score
        dlen = rec["doc_len"]
        norm = 1.0 - b + b * (dlen / avgdl)
        for tl, (orig_tok, w) in q_lower_to_orig.items():
            tf_lower = rec["lower_counts"].get(tl, 0)
            tf_exact = rec["exact_counts"].get(orig_tok, 0)
            if tf_lower > 0:
                matched_toks.append(orig_tok)
                # Exact case match gets full weight; case-only match gets 0.75x
                eff_tf = tf_exact + 0.75 * (tf_lower - tf_exact)
                tf_sat = (eff_tf * (k1 + 1.0)) / (eff_tf + k1 * norm)
                score += w * idf[tl] * tf_sat

            # Path/module component match bonus
            if tl == stem_lower and stem_lower not in GENERIC_STEMS:
                score += 4.5 * w * max(1.0, idf[tl])
            elif tl in path_parts_lower and tl not in GENERIC_STEMS:
                score += 2.2 * w * max(1.0, idf[tl])

        if score <= 0.0:
            continue

        is_docs_src = rel_p.startswith("docs_src/")
        if is_docs_src:
            score *= 1.20 if wants_docs_src else 0.75
        else:
            score *= 1.20

        rec["prelim_score"] = score
        rec["matched_tokens"] = matched_toks
        prelim.append(rec)

    prelim.sort(key=lambda r: -r["prelim_score"])

    # AST parse & snippet extraction on top 25 candidates only (<40ms)
    top_pool = prelim[:25]
    top_tok_regex = None
    if q_tokens:
        # Match highest-weight tokens for line snippet extraction
        pat_toks = [re.escape(t) for t, _ in sorted(q_tokens.items(), key=lambda kv: -kv[1])[:16]]
        top_tok_regex = re.compile(r"\b(?:" + "|".join(pat_toks) + r")\b", re.IGNORECASE)

    final_candidates: list[dict[str, Any]] = []
    for rec in top_pool:
        content = rec["content"]
        lines = content.splitlines()
        matched_line_nums: list[int] = []
        snippets: list[dict[str, Any]] = []

        if top_tok_regex is not None:
            for idx, ln_txt in enumerate(lines, 1):
                if top_tok_regex.search(ln_txt):
                    matched_line_nums.append(idx)
                    if len(snippets) < 3:
                        snippets.append({"line": idx, "text": ln_txt.strip()[:100]})

        ast_bonus, ast_syms = _extract_ast_symbols_and_enclosing(
            content, q_tokens, matched_line_nums[:20]
        )
        total_score = round(rec["prelim_score"] + ast_bonus, 2)
        final_candidates.append({
            "file": rec["rel"],
            "score": total_score,
            "total_lines": len(lines),
            "matched_tokens": rec["matched_tokens"][:8],
            "symbols": ast_syms,
            "snippets": snippets,
        })

    final_candidates.sort(key=lambda c: -c["score"])
    chosen = final_candidates[:top_k]

    recommended_tests = find_relevant_tests(
        workspace=workspace,
        test_files=test_files,
        top_candidates=chosen[:3],
        query_tokens=q_tokens,
    )

    return {
        "status": "ok",
        "search_tokens": list(q_tokens.keys())[:12],
        "top_candidates": chosen,
        "recommended_tests": recommended_tests,
    }


def find_relevant_tests(
    workspace: Path,
    test_files: list[tuple[Path, str]],
    top_candidates: list[dict[str, Any]],
    query_tokens: dict[str, float] | None = None,
    max_tests: int = 5,
) -> list[str]:
    if not test_files or not top_candidates:
        return []

    target_modules: list[str] = []
    specific_stems: set[str] = set()
    parent_Dirs: set[str] = set()
    symbol_names: set[str] = set()

    for rank_idx, cand in enumerate(top_candidates):
        rel_p = cand["file"]
        parts = list(Path(rel_p).with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts:
            mod_dotted = ".".join(parts)
            target_modules.append(mod_dotted)
            if len(parts) >= 2:
                target_modules.append(".".join(parts[:-1]))
            stem = parts[-1].lower()
            if stem not in GENERIC_STEMS:
                specific_stems.add(stem)
            for parent_part in parts[1:-1]:
                pl = parent_part.lower()
                if pl not in GENERIC_STEMS:
                    parent_Dirs.add(pl)
        for sym in cand.get("symbols", []):
            short = sym["name"].split(".")[-1]
            if len(short) >= 4 and short.lower() not in GENERIC_STEMS:
                symbol_names.add(short)

    q_specific = {
        t.lower() for t, w in (query_tokens or {}).items()
        if w >= 1.5 and len(t) >= 4 and t.lower() not in GENERIC_STEMS
    }

    scored_tests: list[tuple[float, str]] = []
    for full_p, rel_t in test_files:
        t_score = 0.0
        t_name_low = Path(rel_t).stem.lower()
        rel_t_low = rel_t.lower()

        for st in specific_stems:
            if t_name_low in {f"test_{st}", f"{st}_test"}:
                t_score += 28.0
            elif st in t_name_low or st.replace("_", "") in t_name_low.replace("_", ""):
                t_score += 14.0

        for pd in parent_Dirs:
            if f"test_{pd}" in t_name_low or f"/{pd}/" in rel_t_low:
                t_score += 12.0

        for qt in q_specific:
            if qt in t_name_low:
                t_score += 8.0

        # Lightweight content check for imports & target symbols
        try:
            t_content = full_p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            t_content = ""

        if t_content:
            for idx_m, mod in enumerate(target_modules):
                if mod in t_content:
                    t_score += 18.0 if idx_m == 0 else 8.0
            for sym in symbol_names:
                if sym in t_content:
                    t_score += 9.0
            for qt in q_specific:
                if qt in t_content.lower():
                    t_score += 2.0

        if t_score > 0:
            scored_tests.append((t_score, rel_t))

    scored_tests.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return [rel_t for _, rel_t in scored_tests[:max_tests]]


def main(raw_argv: list[str] | None = None) -> int:
    argv = _normalize_argv(sys.argv[1:] if raw_argv is None else raw_argv)
    parser = argparse.ArgumentParser(description="BM25 + AST Code & Test Locator")
    parser.add_argument("positional_query", nargs="*", help="Query string or keywords")
    parser.add_argument("--query", "-q", default=None, help="Problem statement or search keywords")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root (default: /workspace or .)")
    parser.add_argument("--top-k", "-k", type=int, default=6, help="Max candidate files to return")
    args = parser.parse_args(argv)

    # No stdin fallback: under run_skill_script stdin can be an open pipe that never closes, and reading it would
    # hang until the harness's per-command timeout (240 s of the task's 8 minutes).
    query = args.query or " ".join(args.positional_query).strip()

    if not query:
        print(json.dumps({"status": "error", "message": "Provide --query '<problem keywords>'"}))
        return 1

    default_ws = Path("/workspace") if Path("/workspace/.git").exists() else Path(".")
    ws = Path(args.workspace).resolve() if args.workspace else default_ws.resolve()

    res = rank_workspace(ws, query=query, top_k=max(1, min(10, args.top_k)))
    out = json.dumps(res, indent=2)
    if len(out) > 4500:
        # Compact fallback to ensure output never hits harness 5,000-char stdout truncation
        out = json.dumps(res, separators=(",", ":"))[:4500]
    print(out)
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
