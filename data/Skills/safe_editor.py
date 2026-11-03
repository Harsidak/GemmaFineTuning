#!/usr/bin/env python3
"""Safe Editor Skill: Resilient string replacement with AST syntax validation & automatic rollback.

Guarantees edits never introduce syntax errors. Verifies exact or flexible line matching
before modifying code, keeping diffs surgical and clean.

Usage:
    python safe_editor.py --file fastapi/routing.py --old "def foo():" --new "def foo(x=None):"
    python safe_editor.py --file rich/console.py --old-file /tmp/old.txt --new-file /tmp/new.txt
"""

import argparse
import ast
import difflib
import json
import sys
from pathlib import Path


def perform_edit(filepath: Path, old_str: str, new_str: str, dry_run: bool = False) -> dict:
    if not filepath.is_file():
        return {"status": "error", "error_type": "FileNotFound", "message": f"File {filepath} does not exist"}

    original = filepath.read_text(encoding="utf-8", errors="ignore")
    if not old_str:
        return {"status": "error", "error_type": "InvalidInput", "message": "old_string cannot be empty"}

    # Pass 1: Exact match
    count = original.count(old_str)
    strategy = "exact"
    replaced = None

    if count == 1:
        replaced = original.replace(old_str, new_str, 1)
    elif count > 1:
        return {
            "status": "error",
            "error_type": "MultipleOccurrences",
            "message": f"old_string matches {count} occurrences in {filepath}. Provide more surrounding context."
        }
    else:
        # Pass 2: Line-by-line whitespace-flexible match
        old_lines = [l.strip() for l in old_str.splitlines() if l.strip()]
        orig_lines = original.splitlines(keepends=True)
        matched_idx = -1

        for i in range(len(orig_lines) - len(old_lines) + 1):
            window = [orig_lines[i + j].strip() for j in range(len(old_lines))]
            if window == old_lines:
                if matched_idx != -1:
                    return {
                        "status": "error",
                        "error_type": "MultipleOccurrences",
                        "message": "Whitespace-flexible matching found multiple candidate locations."
                    }
                matched_idx = i

        if matched_idx != -1:
            strategy = "flexible_whitespace"
            # Replace lines matched_idx .. matched_idx + len(old_lines)
            prefix = "".join(orig_lines[:matched_idx])
            suffix = "".join(orig_lines[matched_idx + len(old_lines):])
            replaced = prefix + new_str + ("\n" if not new_str.endswith("\n") else "") + suffix
        else:
            return {
                "status": "error",
                "error_type": "MatchNotFound",
                "message": f"old_string could not be located in {filepath}"
            }

    # AST syntax check on Python files
    if filepath.suffix == ".py":
        try:
            ast.parse(replaced, filename=str(filepath))
        except SyntaxError as syn_err:
            return {
                "status": "error",
                "error_type": "SyntaxError",
                "message": f"Replacement introduced SyntaxError at line {syn_err.lineno}: {syn_err.msg}",
                "syntax_error_line": syn_err.lineno,
                "syntax_error_offset": syn_err.offset,
                "text": syn_err.text
            }

    # Generate diff
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        replaced.splitlines(keepends=True),
        fromfile=f"a/{filepath.name}",
        tofile=f"b/{filepath.name}",
        n=3
    ))
    diff_text = "".join(diff_lines)

    if not dry_run:
        filepath.write_text(replaced, encoding="utf-8")

    return {
        "status": "ok",
        "strategy": strategy,
        "dry_run": dry_run,
        "file": str(filepath),
        "lines_changed": len(diff_lines),
        "diff": diff_text[:4000]
    }


def main():
    parser = argparse.ArgumentParser(description="Safe AST-Validated File Editor")
    parser.add_argument("--file", "-f", required=True, help="Path to file to edit")
    parser.add_argument("--old", default=None, help="Exact code to replace")
    parser.add_argument("--new", default=None, help="New code replacement")
    parser.add_argument("--old-file", default=None, help="File containing old string")
    parser.add_argument("--new-file", default=None, help="File containing new string")
    parser.add_argument("--dry-run", action="store_true", help="Check replacement and diff without modifying file")
    args = parser.parse_args()

    old_str = args.old
    if not old_str and args.old_file:
        old_str = Path(args.old_file).read_text(encoding="utf-8")

    new_str = args.new
    if not new_str and args.new_file:
        new_str = Path(args.new_file).read_text(encoding="utf-8")

    if old_str is None or new_str is None:
        print(json.dumps({"status": "error", "message": "Must provide both old and new strings"}))
        sys.exit(1)

    res = perform_edit(Path(args.file), old_str, new_str, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))
    if res.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
