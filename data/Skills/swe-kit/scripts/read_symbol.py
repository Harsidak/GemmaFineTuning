#!/usr/bin/env python3
"""AST Symbol & Enclosing-Scope Reader for SWE-bench repositories (stdlib only).

Reads exact function, async function, class, or method spans with 1-indexed line
numbers so edit_file has exact whitespace and context without 150-line blind slicing.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import sys
import warnings
from pathlib import Path
from typing import Any

MAX_CODE_LINES = 120
MAX_CODE_CHARS = 3600


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
                    for key, flag in (
                        ("file", "--file"),
                        ("path", "--file"),
                        ("file_path", "--file"),
                        ("symbol", "--symbol"),
                        ("line", "--line"),
                        ("range", "--range"),
                        ("context", "--context"),
                        ("workspace", "--workspace"),
                    ):
                        if key in obj and obj[key] is not None:
                            expanded.extend([flag, str(obj[key])])
                    if obj.get("outline") or obj.get("list_symbols"):
                        expanded.append("--outline")
                    return expanded
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if single.startswith("-") and " " in single:
            try:
                return shlex.split(single)
            except ValueError:
                pass
    return raw_argv


def collect_symbols(content: str) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(content)
    symbols: list[dict[str, Any]] = []

    def _visit(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                full_name = f"{prefix}.{child.name}" if prefix else child.name
                if isinstance(child, ast.ClassDef):
                    kind = "class"
                elif isinstance(child, ast.AsyncFunctionDef):
                    kind = "async_def"
                else:
                    kind = "def"
                # Include decorators in start_line so edits around decorators never miss lines
                deco_lines = [getattr(d, "lineno", child.lineno) for d in getattr(child, "decorator_list", [])]
                start_ln = min([child.lineno, *deco_lines])
                end_ln = getattr(child, "end_lineno", child.lineno) or child.lineno
                symbols.append({
                    "name": full_name,
                    "short_name": child.name,
                    "kind": kind,
                    "start_line": start_ln,
                    "def_line": child.lineno,
                    "end_line": end_ln,
                    "line_count": end_ln - start_ln + 1,
                })
                if isinstance(child, ast.ClassDef):
                    _visit(child, full_name)

    _visit(tree)
    return symbols


def _format_numbered_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    focus_line: int | None = None,
) -> tuple[str, bool]:
    start = max(1, start_line)
    end = min(len(lines), end_line)
    total = end - start + 1
    truncated = False

    if total > MAX_CODE_LINES:
        truncated = True
        head_n = 75
        tail_n = 35
        head_part = [
            f"{idx:5d}{'*' if focus_line == idx else ' '} | {lines[idx - 1]}"
            for idx in range(start, start + head_n)
        ]
        omitted_start = start + head_n
        omitted_end = end - tail_n
        tail_part = [
            f"{idx:5d}{'*' if focus_line == idx else ' '} | {lines[idx - 1]}"
            for idx in range(end - tail_n + 1, end + 1)
        ]
        rendered = (
            "\n".join(head_part)
            + f"\n  ... [lines {omitted_start}-{omitted_end} omitted; pass --range {omitted_start}-{min(omitted_end, omitted_start + 80)} to inspect] ...\n"
            + "\n".join(tail_part)
        )
    else:
        rendered = "\n".join(
            f"{idx:5d}{'*' if focus_line == idx else ' '} | {lines[idx - 1]}"
            for idx in range(start, end + 1)
        )

    if len(rendered) > MAX_CODE_CHARS:
        truncated = True
        rendered = rendered[:MAX_CODE_CHARS] + "\n  ... [truncated at char cap; use --range start-end]"
    return rendered, truncated


def resolve_target_file(workspace: Path, file_arg: str) -> Path:
    cand = Path(file_arg)
    if cand.is_file():
        return cand
    ws_cand = workspace / file_arg.lstrip("/")
    if ws_cand.is_file():
        return ws_cand
    # Strip /workspace/ prefix if workspace is a custom local path
    if file_arg.startswith("/workspace/"):
        rel = file_arg[len("/workspace/"):]
        if (workspace / rel).is_file():
            return workspace / rel
    return ws_cand


def inspect_file(
    workspace: Path,
    file_arg: str,
    symbol: str | None = None,
    line: int | None = None,
    line_range: str | None = None,
    context: int = 20,
    outline: bool = False,
) -> dict[str, Any]:
    # Support shorthand `pkg/mod.py:SymbolName` or `pkg/mod.py:100-140` or `pkg/mod.py:120`
    if ":" in file_arg and not Path(file_arg).drive:
        lhs, rhs = file_arg.split(":", 1)
        file_arg = lhs
        if rhs and symbol is None and line is None and line_range is None:
            if "-" in rhs and all(p.isdigit() for p in rhs.split("-", 1)):
                line_range = rhs
            elif rhs.isdigit():
                line = int(rhs)
            else:
                symbol = rhs

    target_path = resolve_target_file(workspace, file_arg)
    if not target_path.is_file():
        return {"status": "error", "message": f"File not found: {file_arg}"}

    try:
        content = target_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"status": "error", "message": f"Cannot read {file_arg}: {exc}"}

    lines = content.splitlines()
    try:
        rel_display = target_path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        rel_display = target_path.as_posix()

    symbols: list[dict[str, Any]] = []
    syntax_err: str | None = None
    try:
        symbols = collect_symbols(content)
    except SyntaxError as exc:
        syntax_err = f"Line {exc.lineno}: {exc.msg}"

    if line_range:
        m_parts = line_range.replace(":", "-").split("-", 1)
        if len(m_parts) == 2 and m_parts[0].strip().isdigit() and m_parts[1].strip().isdigit():
            s_ln = max(1, int(m_parts[0].strip()))
            e_ln = min(len(lines), int(m_parts[1].strip()))
            if e_ln < s_ln:
                s_ln, e_ln = e_ln, s_ln
            code, trunc = _format_numbered_span(lines, s_ln, e_ln)
            return {
                "status": "ok",
                "file": rel_display,
                "start_line": s_ln,
                "end_line": e_ln,
                "total_lines": len(lines),
                "truncated": trunc,
                "code": code,
            }
        return {"status": "error", "message": f"Invalid --range '{line_range}'; expected START-END"}

    if outline:
        compact_syms = [
            {"name": s["name"], "kind": s["kind"], "lines": [s["start_line"], s["end_line"]]}
            for s in symbols[:60]
        ]
        res: dict[str, Any] = {
            "status": "ok",
            "file": rel_display,
            "total_lines": len(lines),
            "symbol_count": len(symbols),
            "symbols": compact_syms,
        }
        if syntax_err:
            res["syntax_error"] = syntax_err
        return res

    if symbol:
        if syntax_err and not symbols:
            return {"status": "error", "file": rel_display, "message": f"SyntaxError prevents AST lookup: {syntax_err}"}
        q_exact = symbol.strip()
        q_low = q_exact.lower()
        exact = [s for s in symbols if s["name"] == q_exact or s["short_name"] == q_exact]
        ci = [s for s in symbols if s["name"].lower() == q_low or s["short_name"].lower() == q_low]
        suffix = [s for s in symbols if s["name"].lower().endswith("." + q_low)]
        sub = [s for s in symbols if q_low in s["name"].lower()]
        matches = exact or ci or suffix or sub

        if not matches:
            return {
                "status": "error",
                "file": rel_display,
                "message": f"Symbol '{symbol}' not found in {rel_display}",
                "available_symbols": [s["name"] for s in symbols[:30]],
            }

        # Prefer functions/methods over giant class bodies when short name collides, unless class was requested
        chosen = matches[0]
        other_matches = [
            {"name": m["name"], "lines": [m["start_line"], m["end_line"]]}
            for m in matches[1:5]
        ]
        code, trunc = _format_numbered_span(lines, chosen["start_line"], chosen["end_line"])
        out: dict[str, Any] = {
            "status": "ok",
            "file": rel_display,
            "symbol": chosen["name"],
            "kind": chosen["kind"],
            "start_line": chosen["start_line"],
            "end_line": chosen["end_line"],
            "total_lines": len(lines),
            "truncated": trunc,
            "code": code,
        }
        if other_matches:
            out["other_matches"] = other_matches
        return out

    if line is not None:
        target_ln = max(1, min(len(lines) if lines else 1, line))
        enclosing = [s for s in symbols if s["start_line"] <= target_ln <= s["end_line"]]
        # Prefer innermost function/method if <= 90 lines; otherwise show window around target_ln
        innermost = enclosing[-1] if enclosing else None
        if innermost and innermost["kind"] != "class" and innermost["line_count"] <= 90:
            s_ln, e_ln = innermost["start_line"], innermost["end_line"]
        else:
            s_ln = max(1, target_ln - context)
            e_ln = min(len(lines), target_ln + context)
        code, trunc = _format_numbered_span(lines, s_ln, e_ln, focus_line=target_ln)
        return {
            "status": "ok",
            "file": rel_display,
            "target_line": target_ln,
            "enclosing_symbol": innermost["name"] if innermost else None,
            "enclosing_lines": [innermost["start_line"], innermost["end_line"]] if innermost else None,
            "start_line": s_ln,
            "end_line": e_ln,
            "truncated": trunc,
            "code": code,
        }

    # Default when only --file is passed: return outline (or full code if short file <= 90 lines)
    if len(lines) <= 90:
        code, trunc = _format_numbered_span(lines, 1, len(lines))
        return {
            "status": "ok",
            "file": rel_display,
            "total_lines": len(lines),
            "truncated": trunc,
            "code": code,
        }
    return {
        "status": "ok",
        "file": rel_display,
        "total_lines": len(lines),
        "symbol_count": len(symbols),
        "symbols": [
            {"name": s["name"], "kind": s["kind"], "lines": [s["start_line"], s["end_line"]]}
            for s in symbols[:45]
        ],
    }


def main(raw_argv: list[str] | None = None) -> int:
    argv = _normalize_argv(sys.argv[1:] if raw_argv is None else raw_argv)
    parser = argparse.ArgumentParser(description="AST Symbol & Scope Reader")
    parser.add_argument("positional", nargs="*", help="Optional positional file[:symbol_or_range] [symbol]")
    parser.add_argument("--file", "-f", default=None, help="Relative path in /workspace")
    parser.add_argument("--symbol", "-s", default=None, help="Function, class, or Class.method name")
    parser.add_argument("--line", "-l", type=int, default=None, help="1-indexed line number to inspect")
    parser.add_argument("--range", "-r", dest="line_range", default=None, help="Line range START-END")
    parser.add_argument("--context", "-c", type=int, default=20, help="Lines of context around --line")
    parser.add_argument("--outline", "--list-symbols", action="store_true", help="List AST symbols and line spans")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root (default: /workspace or .)")
    args = parser.parse_args(argv)

    file_arg = args.file
    symbol = args.symbol
    if not file_arg and args.positional:
        file_arg = args.positional[0]
        if len(args.positional) >= 2 and symbol is None:
            symbol = args.positional[1]

    if not file_arg:
        print(json.dumps({"status": "error", "message": "Provide --file <path> [--symbol <name> | --line <N> | --outline]"}))
        return 1

    default_ws = Path("/workspace") if Path("/workspace/.git").exists() else Path(".")
    ws = Path(args.workspace).resolve() if args.workspace else default_ws.resolve()

    res = inspect_file(
        workspace=ws,
        file_arg=file_arg,
        symbol=symbol,
        line=args.line,
        line_range=args.line_range,
        context=max(5, min(60, args.context)),
        outline=args.outline,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
