#!/usr/bin/env python3
"""Code Locator Skill: Fast, deterministic symbol, file, and test locator for SWE-bench repositories.

Uses pure Python standard library (ast, os, re, sys, pathlib) so it runs offline in any sandbox.
Solves the #1 SWE-bench failure mode: statement mentions symptoms/tokens but NOT the exact file.

Usage:
    python code_locator.py --query "proxy isatty" --workspace /workspace
    python code_locator.py --query "serialize_response status_code" --workspace . --top-k 5
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

# Directories to ignore during source search
IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env",
    "build", "dist", "site-packages", ".mypy_cache", ".ruff_cache",
    "docs", "docs_src", "examples", "tutorial"
}

def extract_tokens(query: str) -> list[str]:
    """Extracts code-like identifiers and keywords from natural language problem statement."""
    # Strip URLs, issue links, markdown formatting
    cleaned = re.sub(r"https?://\S+", " ", query)
    cleaned = re.sub(r"Fixes\s+#?\d+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[`'\"*#_]", " ", cleaned)
    
    # Extract identifiers (snake_case, camelCase, dotted names)
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    
    # Filter stopwords
    stopwords = {
        "the", "and", "is", "in", "to", "for", "with", "that", "this", "from",
        "as", "at", "by", "on", "if", "not", "be", "an", "or", "it", "are",
        "when", "using", "use", "support", "add", "fix", "issue", "problem",
        "error", "bug", "fails", "fail", "should", "would", "can", "could"
    }
    tokens = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if len(tok) >= 3 and tok_lower not in stopwords:
            tokens.append(tok)
            # If camelCase or snake_case, also add sub-parts
            subparts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", tok)
            if len(subparts) > 1:
                for sp in subparts:
                    if len(sp) >= 3 and sp.lower() not in stopwords:
                        tokens.append(sp)
    return list(dict.fromkeys(tokens))  # preserve order, deduplicate


def scan_py_file(path: Path, tokens: list[str]) -> dict:
    """Scans a python file for token occurrences and AST definitions."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    score = 0
    matched_tokens = set()
    matched_lines = []
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        for tok in tokens:
            if re.search(r"\b" + re.escape(tok) + r"\b", line, re.IGNORECASE):
                matched_tokens.add(tok)
                score += (3 if tok in line else 1)
                if len(matched_lines) < 8:
                    matched_lines.append({"line": i, "text": line.strip()[:120]})

    if not matched_tokens:
        return None

    # AST symbol extraction for definitions
    symbols = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for tok in tokens:
                    if tok.lower() in node.name.lower():
                        symbols.append({
                            "type": type(node).__name__,
                            "name": node.name,
                            "line": node.lineno,
                            "end_line": getattr(node, "end_lineno", node.lineno)
                        })
                        score += 15  # Heavy weight for symbol definition match
    except Exception:
        pass

    return {
        "file": str(path),
        "score": score,
        "token_coverage": f"{len(matched_tokens)}/{len(tokens)}",
        "matched_tokens": sorted(matched_tokens),
        "symbols": symbols[:5],
        "line_snippets": matched_lines[:5],
        "total_lines": len(lines)
    }


def find_candidate_files(workspace: Path, tokens: list[str], top_k: int = 5) -> list[dict]:
    """Finds top candidate source files and corresponding test files."""
    candidates = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py") and not f.startswith("."):
                p = Path(root) / f
                # Prioritize source files over test files initially
                rel_path = p.relative_to(workspace).as_posix()
                is_test = "tests/" in rel_path or "test_" in f or "_test.py" in f
                res = scan_py_file(p, tokens)
                if res:
                    res["is_test"] = is_test
                    res["rel_file"] = rel_path
                    # Modulate score: source code matches preferred over test fixtures for code changes
                    if not is_test:
                        res["score"] *= 1.5
                    candidates.append(res)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def find_relevant_tests(workspace: Path, source_files: list[str]) -> list[str]:
    """Heuristically finds tests covering the target source files."""
    tests = set()
    tests_dir = workspace / "tests"
    if not tests_dir.is_dir():
        return []

    for sf in source_files:
        stem = Path(sf).stem
        # e.g., fastapi/routing.py -> tests/test_routing.py or test_*routing*.py
        for root, _, files in os.walk(tests_dir):
            for f in files:
                if f.endswith(".py") and (stem in f or stem.replace("_", "") in f.replace("_", "")):
                    tests.add((Path(root) / f).relative_to(workspace).as_posix())
    return sorted(tests)[:6]


def main():
    parser = argparse.ArgumentParser(description="Deterministic Code & Symbol Locator")
    parser.add_argument("--query", "-q", required=True, help="Bug/PR problem statement or search string")
    parser.add_argument("--workspace", "-w", default=".", help="Path to repository workspace root")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of candidate files to return")
    parser.add_argument("--json", action="store_true", default=True, help="Output JSON format")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    tokens = extract_tokens(args.query)
    if not tokens:
        print(json.dumps({"status": "error", "message": "No meaningful search tokens found in query"}))
        return

    candidates = find_candidate_files(ws, tokens, top_k=args.top_k)
    source_candidates = [c["rel_file"] for c in candidates if not c.get("is_test")]
    tests = find_relevant_tests(ws, source_candidates)

    output = {
        "status": "ok",
        "search_tokens": tokens[:12],
        "top_candidates": candidates,
        "recommended_tests_to_run": tests
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
