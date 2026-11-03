#!/usr/bin/env python3
"""AST Reader Skill: Precise, windowed function and class inspection for large Python files.

Solves the large file dilemma in SWE-bench (files are 1,000–4,500 lines, read_file limit is 150 lines).
Extracts exact start/end line bounds and source definitions of target functions or classes.

Usage:
    python ast_reader.py --file fastapi/routing.py --symbol serialize_response
    python ast_reader.py --file rich/console.py --line 512 --context 20
    python ast_reader.py --file fastapi/applications.py --list-symbols
"""

import argparse
import ast
import json
import sys
from pathlib import Path


def find_symbols_in_file(content: str) -> list[dict]:
    """Extracts all top-level and method definitions with exact line ranges."""
    tree = ast.parse(content)
    symbols = []

    def _traverse(node, parent_name=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                full_name = f"{parent_name}.{child.name}" if parent_name else child.name
                kind = "class" if isinstance(child, ast.ClassDef) else ("async_def" if isinstance(child, ast.AsyncFunctionDef) else "def")
                symbols.append({
                    "name": full_name,
                    "short_name": child.name,
                    "kind": kind,
                    "start_line": child.lineno,
                    "end_line": getattr(child, "end_lineno", child.lineno),
                    "lines_count": getattr(child, "end_lineno", child.lineno) - child.lineno + 1
                })
                if isinstance(child, ast.ClassDef):
                    _traverse(child, full_name)

    _traverse(tree)
    return symbols


def extract_symbol_code(lines: list[str], symbol_info: dict) -> str:
    """Formats the symbol code with 1-indexed line numbers."""
    start = max(1, symbol_info["start_line"])
    end = min(len(lines), symbol_info["end_line"])
    numbered_lines = []
    for idx in range(start - 1, end):
        numbered_lines.append(f"{idx + 1:5d} | {lines[idx]}")
    return "\n".join(numbered_lines)


def main():
    parser = argparse.ArgumentParser(description="AST Code Scope & Symbol Inspector")
    parser.add_argument("--file", "-f", required=True, help="Path to Python source file")
    parser.add_argument("--symbol", "-s", default=None, help="Symbol name to inspect (class, method, or function)")
    parser.add_argument("--line", "-l", type=int, default=None, help="Line number to inspect enclosing scope for")
    parser.add_argument("--context", "-c", type=int, default=15, help="Context lines around target line")
    parser.add_argument("--list-symbols", action="store_true", help="List all symbols and their line ranges")
    args = parser.parse_args()

    p = Path(args.file)
    if not p.is_file():
        print(json.dumps({"status": "error", "message": f"File not found: {args.file}"}))
        sys.exit(1)

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(json.dumps({"status": "error", "message": f"Cannot read file: {exc}"}))
        sys.exit(1)

    lines = content.splitlines()

    try:
        symbols = find_symbols_in_file(content)
    except Exception as exc:
        print(json.dumps({"status": "error", "message": f"AST parse error: {exc}"}))
        sys.exit(1)

    if args.list_symbols:
        print(json.dumps({
            "status": "ok",
            "file": str(p),
            "total_lines": len(lines),
            "symbol_count": len(symbols),
            "symbols": symbols
        }, indent=2))
        return

    if args.symbol:
        sym_lower = args.symbol.lower()
        matched = [s for s in symbols if s["short_name"].lower() == sym_lower or s["name"].lower() == sym_lower]
        if not matched:
            # Substring match fallback
            matched = [s for s in symbols if sym_lower in s["name"].lower()]

        if not matched:
            print(json.dumps({
                "status": "error",
                "message": f"Symbol '{args.symbol}' not found in {args.file}",
                "available_symbols": [s["name"] for s in symbols[:20]]
            }, indent=2))
            return

        target = matched[0]
        code = extract_symbol_code(lines, target)
        print(json.dumps({
            "status": "ok",
            "file": str(p),
            "symbol": target["name"],
            "kind": target["kind"],
            "start_line": target["start_line"],
            "end_line": target["end_line"],
            "lines_count": target["lines_count"],
            "code": code
        }, indent=2))
        return

    if args.line is not None:
        target_line = args.line
        enclosing = [s for s in symbols if s["start_line"] <= target_line <= s["end_line"]]
        enclosing_symbol = enclosing[-1] if enclosing else None

        start = max(1, target_line - args.context)
        end = min(len(lines), target_line + args.context)
        numbered_lines = [f"{idx + 1:5d}{'*' if idx + 1 == target_line else ' '} | {lines[idx]}" for idx in range(start - 1, end)]

        print(json.dumps({
            "status": "ok",
            "file": str(p),
            "target_line": target_line,
            "enclosing_symbol": enclosing_symbol["name"] if enclosing_symbol else None,
            "enclosing_range": [enclosing_symbol["start_line"], enclosing_symbol["end_line"]] if enclosing_symbol else None,
            "snippet": "\n".join(numbered_lines)
        }, indent=2))
        return

    # Default: summary of file
    print(json.dumps({
        "status": "ok",
        "file": str(p),
        "total_lines": len(lines),
        "symbols_count": len(symbols),
        "top_symbols": symbols[:15]
    }, indent=2))


if __name__ == "__main__":
    main()
