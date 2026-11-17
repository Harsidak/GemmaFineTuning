#!/usr/bin/env python3
"""Pre-Submit Patch Auditor & Targeted Pytest Runner (stdlib only).

Validates workspace diffs against exact SWEGemma grading rules BEFORE submit_patch():
1. Detects untracked scratch script leaks BEFORE `git add -N .` (and via base-commit diff).
2. Simulates HARNESS_README §8.1 protected-file stripping (`tests/`, `conftest.py`,
   `pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`, `sitecustomize.py`, `*.pth`)
   and HARD-BLOCKS when `kept_source_files` would be empty or tests were tampered with.
3. Runs `ast.parse` on every modified `.py` source file.
4. Auto-selects and runs targeted `pytest -x -q` with a hard timeout unless `--no-tests`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import warnings as _warnings
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from locate import _collect_repo_py_files, find_relevant_tests

HARNESS_PROTECTED_CONFIGS = {
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
}

HARNESS_STARTUP_HOOKS = {
    "sitecustomize.py",
    "usercustomize.py",
    "_swegemma_stubs.py",
}

SCRATCH_NAME_RE = re.compile(
    r"(?:^|_)(?:repro|reproduce|debug|scratch|tmp|temp|playground)(?:_|\b|\.)",
    re.IGNORECASE,
)
SCRATCH_SUFFIXES = {".patch", ".orig", ".bak", ".rej", ".pyc", ".log", ".out"}


def _normalize_argv(raw_argv: list[str]) -> list[str]:
    if not raw_argv:
        env_args = os.environ.get("SKILL_ARGS") or os.environ.get("ARGS") or ""
        if env_args.strip():
            raw_argv = [env_args.strip()]
        else:
            return []

    if len(raw_argv) == 1:
        single = raw_argv[0].strip()
        if not single or single in {"{}", "[]"}:
            return []
        if single.startswith("{") and single.endswith("}"):
            try:
                obj = json.loads(single)
                if isinstance(obj, dict):
                    expanded: list[str] = []
                    if obj.get("workspace"):
                        expanded.extend(["--workspace", str(obj["workspace"])])
                    tf = obj.get("test_file") or obj.get("test_files") or obj.get("tests")
                    if isinstance(tf, list):
                        for item in tf:
                            expanded.extend(["--test-file", str(item)])
                    elif isinstance(tf, str) and tf.strip():
                        expanded.extend(["--test-file", tf.strip()])
                    kw = obj.get("keyword") or obj.get("k")
                    if kw:
                        expanded.extend(["-k", str(kw)])
                    if obj.get("no_tests") or obj.get("skip_tests"):
                        expanded.append("--no-tests")
                    if "timeout" in obj and obj["timeout"] is not None:
                        expanded.extend(["--timeout", str(obj["timeout"])])
                    return expanded
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if single.startswith("-") and " " in single:
            try:
                return shlex.split(single)
            except ValueError:
                pass
    return raw_argv


def is_harness_stripped(rel_posix: str) -> tuple[bool, str]:
    """Exact mirror of HARNESS_README.md §8.1 protected-file reset/strip rules."""
    norm = rel_posix.strip().lstrip("./")
    parts = norm.split("/")
    basename = parts[-1]

    if basename in HARNESS_STARTUP_HOOKS or basename.endswith(".pth"):
        return True, "startup_hook"
    if basename in HARNESS_PROTECTED_CONFIGS:
        return True, "runner_config"
    if basename == "conftest.py":
        return True, "test_file"
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return True, "test_file"
    if any(p in {"tests", "test", "testing"} for p in parts[:-1]):
        return True, "test_file"
    return False, ""


def is_stray_scratch_file(rel_posix: str, repo_has_root_py: bool) -> bool:
    norm = rel_posix.strip().lstrip("./")
    parts = norm.split("/")
    basename = parts[-1]
    suffix = Path(basename).suffix.lower()

    if suffix in SCRATCH_SUFFIXES or basename.startswith("."):
        return True
    if SCRATCH_NAME_RE.search(basename):
        return True
    # Any newly created file directly in /workspace root (when the repo lives in package subdirs) is a scratch leak
    return len(parts) == 1 and not repo_has_root_py


def _git(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _detect_base_ref(workspace: Path) -> str:
    chk = _git(workspace, ["rev-parse", "--verify", "--quiet", "_swegemma_baseline"])
    if chk.returncode == 0 and chk.stdout.strip():
        return "_swegemma_baseline"
    return "HEAD"


def _repo_tracks_root_py(workspace: Path, base_ref: str) -> bool:
    tracked = _git(workspace, ["ls-tree", "--name-only", base_ref])
    if tracked.returncode != 0:
        return False
    ignore_root = {"setup.py", "conftest.py", "noxfile.py", "fabfile.py"}
    for ln in tracked.stdout.splitlines():
        name = ln.strip()
        if name.endswith(".py") and "/" not in name and name not in ignore_root:
            return True
    return False


def _select_pytest_python() -> str:
    venv_py = Path("/opt/venv/bin/python")
    if venv_py.is_file() and os.name != "nt":
        return str(venv_py)
    return sys.executable


def run_targeted_pytest(
    workspace: Path,
    kept_source_files: list[str],
    explicit_tests: list[str] | None = None,
    keyword: str | None = None,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    targets: list[str] = []
    if explicit_tests:
        for t in explicit_tests:
            t_clean = t.strip()
            if not t_clean:
                continue
            if (workspace / t_clean.split("::")[0]).exists():
                targets.append(t_clean)
    else:
        _, test_files = _collect_repo_py_files(workspace)
        cand_dicts = [{"file": f, "symbols": []} for f in kept_source_files[:3]]
        targets = find_relevant_tests(
            workspace=workspace,
            test_files=test_files,
            top_candidates=cand_dicts,
            max_tests=2,
        )

    if not targets:
        return {
            "status": "skipped",
            "targets": [],
            "message": "No matching test files discovered automatically; pass --test-file tests/test_x.py if needed.",
        }

    py_bin = _select_pytest_python()
    cmd = [
        py_bin,
        "-s",
        "-m",
        "pytest",
        *targets,
        "-x",
        "-q",
        "-p",
        "no:anyio",
        "--tb=short",
    ]
    if keyword:
        cmd.extend(["-k", keyword])

    env = os.environ.copy()
    ws_paths = [str(workspace)]
    if (workspace / "src").is_dir():
        ws_paths.append(str(workspace / "src"))
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*ws_paths, *([existing_py_path] if existing_py_path else [])])
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "targets": targets,
            "message": f"pytest timed out after {timeout_sec}s. Narrow with --test-file <single_file> -k <test_name>.",
        }

    combined = (proc.stdout + "\n" + proc.stderr).strip()
    if len(combined) > 1500:
        combined = combined[:500] + "\n...[truncated]...\n" + combined[-950:]

    if proc.returncode == 0:
        return {"status": "passed", "returncode": 0, "targets": targets, "output": combined}
    if proc.returncode == 5:
        # Exit code 5 = no tests collected (e.g. -k didn't match)
        return {
            "status": "no_tests_collected",
            "returncode": 5,
            "targets": targets,
            "output": combined,
        }
    if proc.returncode in {2, 3, 4}:
        # Collection / plugin / usage error: check if caused by an edited source file or module
        edited_markers: set[str] = set()
        for f in kept_source_files:
            edited_markers.add(f)
            mod = f.removesuffix(".py").removeprefix("src/")
            edited_markers.add(mod.replace("/", "."))
        mentions_edited = any(m in combined for m in edited_markers)
        return {
            "status": "failed" if mentions_edited else "collection_warning",
            "returncode": proc.returncode,
            "targets": targets,
            "output": combined,
        }
    return {
        "status": "failed",
        "returncode": proc.returncode,
        "targets": targets,
        "output": combined,
    }


def verify_and_audit(
    workspace: Path,
    explicit_tests: list[str] | None = None,
    keyword: str | None = None,
    no_tests: bool = False,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    if not (workspace / ".git").exists():
        return {"status": "error", "ready_to_submit": False, "message": f"Not a git repository: {workspace}"}

    base_ref = _detect_base_ref(workspace)
    repo_has_root_py = _repo_tracks_root_py(workspace, base_ref)

    # 1. Inspect untracked (`??`) files BEFORE `git add -N .`
    porcelain_before = _git(workspace, ["status", "--porcelain", "-uall"])
    untracked_before: set[str] = set()
    for raw_ln in porcelain_before.stdout.splitlines():
        if raw_ln.startswith("?? "):
            untracked_before.add(raw_ln[3:].strip().strip('"'))

    # 2. Mirror harness `submit_patch` staging (`git add -N .`) so new files appear in diff
    _git(workspace, ["add", "-N", "."])

    # 3. Inspect changed files against baseline ref
    name_status_proc = _git(workspace, ["diff", "--name-status", base_ref])
    diff_proc = _git(workspace, ["diff", "--binary", base_ref])
    raw_diff = diff_proc.stdout

    changed_entries: list[tuple[str, str]] = []
    for ln in name_status_proc.stdout.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        status_code = parts[0].strip()
        rel_p = parts[-1].strip().replace("\\", "/")
        changed_entries.append((status_code, rel_p))

    blocking_errors: list[str] = []
    warnings: list[str] = []
    kept_source_files: list[str] = []
    stripped_files: list[str] = []
    stray_files: list[str] = []
    syntax_checked: list[str] = []

    for status_code, rel_p in changed_entries:
        is_new_file = status_code.startswith("A") or rel_p in untracked_before
        stripped, reason = is_harness_stripped(rel_p)
        if stripped:
            stripped_files.append(rel_p)
            if is_new_file:
                blocking_errors.append(
                    f"New protected {reason} '{rel_p}' is forbidden/stripped by grader. Remove it: rm -f {rel_p}"
                )
            else:
                blocking_errors.append(
                    f"Modified protected {reason} '{rel_p}' will be reverted by grader. Restore it: git checkout -- {rel_p}"
                )
            continue

        if is_new_file and is_stray_scratch_file(rel_p, repo_has_root_py):
            stray_files.append(rel_p)
            blocking_errors.append(
                f"Untracked scratch file '{rel_p}' will pollute git diff. Delete it (or move to /tmp): rm -f {rel_p}"
            )
            continue

        kept_source_files.append(rel_p)

        # Syntax check every existing modified/added .py source file
        full_p = workspace / rel_p
        if rel_p.endswith(".py") and not status_code.startswith("D") and full_p.is_file():
            try:
                src = full_p.read_text(encoding="utf-8", errors="ignore")
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(src, filename=rel_p)
                syntax_checked.append(rel_p)
            except SyntaxError as exc:
                bad_line = (exc.text or "").strip()[:100]
                blocking_errors.append(
                    f"SyntaxError in {rel_p}:{exc.lineno}:{exc.offset or 0}: {exc.msg} ({bad_line})"
                )

    if not changed_entries or not raw_diff.strip():
        blocking_errors.append(
            "git diff is EMPTY. You have not modified any file in /workspace yet."
        )
    elif not kept_source_files and not stray_files:
        blocking_errors.append(
            "Graded source patch is EMPTY (0 kept source files after harness strips protected test/config files). Edit package source code."
        )

    lines_added = sum(1 for ln in raw_diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    lines_deleted = sum(1 for ln in raw_diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))

    if lines_added + lines_deleted > 250:
        warnings.append(
            f"Large diff ({lines_added} insertions, {lines_deleted} deletions). Confirm no unrelated formatting or debug code was left behind."
        )

    test_report: dict[str, Any] = {"status": "skipped", "targets": []}
    if not no_tests and not blocking_errors and kept_source_files:
        test_report = run_targeted_pytest(
            workspace=workspace,
            kept_source_files=kept_source_files,
            explicit_tests=explicit_tests,
            keyword=keyword,
            timeout_sec=timeout_sec,
        )
        t_status = test_report.get("status")
        if t_status == "failed":
            blocking_errors.append(
                f"Targeted pytest failed on {test_report.get('targets')}. Inspect traceback in test_report.output and fix the source."
            )
        elif t_status == "timeout":
            warnings.append(str(test_report.get("message", "pytest timed out")))
        elif t_status == "collection_warning":
            warnings.append(
                "Pre-existing collection error in auto-selected test; verify with --test-file <specific_test.py> if needed."
            )

    ready = len(blocking_errors) == 0 and len(kept_source_files) > 0
    return {
        "status": "ok",
        "ready_to_submit": ready,
        "kept_source_files": kept_source_files,
        "stripped_protected_files": stripped_files,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "syntax_verified_files": syntax_checked,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "test_report": test_report,
    }


def main(raw_argv: list[str] | None = None) -> int:
    argv = _normalize_argv(sys.argv[1:] if raw_argv is None else raw_argv)
    parser = argparse.ArgumentParser(description="Pre-submit patch auditor and targeted pytest runner")
    parser.add_argument("positional_tests", nargs="*", help="Optional test file paths")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root (default: /workspace or .)")
    parser.add_argument("--test-file", "-t", action="append", default=[], help="Specific test file(s) to run")
    parser.add_argument("-k", "--keyword", default=None, help="pytest -k expression")
    parser.add_argument("--no-tests", action="store_true", help="Run diff/syntax/leak audit only without pytest")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for targeted pytest")
    args = parser.parse_args(argv)

    default_ws = Path("/workspace") if Path("/workspace/.git").exists() else Path(".")
    ws = Path(args.workspace).resolve() if args.workspace else default_ws.resolve()

    explicit_tests = [*args.test_file, *args.positional_tests]
    report = verify_and_audit(
        workspace=ws,
        explicit_tests=explicit_tests if explicit_tests else None,
        keyword=args.keyword,
        no_tests=args.no_tests,
        timeout_sec=max(5, min(150, args.timeout)),
    )
    out = json.dumps(report, indent=2)
    if len(out) > 4500:
        out = json.dumps(report, separators=(",", ":"))[:4500]
    print(out)
    # Return exit code 0 for valid audit execution so ADK RunSkillScriptTool always delivers structured JSON
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
