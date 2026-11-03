#!/usr/bin/env python3
"""Patch Auditor Skill: Pre-submit gatekeeper and diff sanitizer.

Checks all anti-patterns and disqualification risks before `submit_patch()` is called:
1. Syntax validity (all modified .py files pass ast.parse).
2. Protected file tampering (warns if tests/ or runner configs were modified).
3. Untracked scratch files (flags scratch scripts that would leak into git diff).
4. Diff sanity (ensures non-empty and surgical).

Usage:
    python patch_auditor.py --workspace /workspace
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PROTECTED_PATTERNS = [
    "conftest.py", "pytest.ini", "pyproject.toml", "tox.ini",
    "setup.cfg", ".pytest.ini", "sitecustomize.py", "usercustomize.py"
]

def audit_workspace(workspace: Path) -> dict:
    if not (workspace / ".git").is_dir():
        return {"status": "error", "message": "Not a git repository"}

    # 1. git add -N . to catch untracked files
    subprocess.run(["git", "add", "-N", "."], cwd=workspace, capture_output=True, text=True)

    # 2. Get git diff
    diff_proc = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True
    )
    diff = diff_proc.stdout

    # 3. Get changed files
    stat_proc = subprocess.run(
        ["git", "diff", "--name-status", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True
    )
    changed_files = [line.strip().split(None, 1) for line in stat_proc.stdout.splitlines() if line.strip()]

    # 4. Check for untracked files
    untracked_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        capture_output=True,
        text=True
    )
    untracked_lines = [l for l in untracked_proc.stdout.splitlines() if l.startswith("??")]

    blocking_errors = []
    warnings = []
    syntax_passed = True
    verified_files = []

    if not diff.strip():
        blocking_errors.append("git diff is completely EMPTY. No changes made.")

    for item in changed_files:
        if len(item) < 2:
            continue
        status_flag, filepath = item[0], item[1]
        p = workspace / filepath

        # Check protected configs
        if any(filepath.endswith(pat) for pat in PROTECTED_PATTERNS):
            warnings.append(f"Protected runner config '{filepath}' modified! Harness will strip this during grading.")

        # Check tests/ modifications
        if filepath.startswith("tests/") or "test_" in filepath or filepath.endswith("_test.py"):
            warnings.append(f"Test file '{filepath}' modified. Harness anti-tampering resets tests before grading.")

        # Check syntax of modified Python files
        if filepath.endswith(".py") and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                ast.parse(content, filename=str(p))
                verified_files.append(filepath)
            except SyntaxError as e:
                syntax_passed = False
                blocking_errors.append(f"SyntaxError in {filepath} (line {e.lineno}): {e.msg}")

    # Check for temporary script leaks in workspace root
    for ul in untracked_lines:
        fname = ul.replace("??", "").strip()
        if fname.endswith(".py") and not fname.startswith("tests/"):
            warnings.append(f"Untracked python file '{fname}' in workspace root! Move to /tmp or delete before submit_patch().")

    # Lines changed math
    added_lines = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deleted_lines = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))

    can_submit = len(blocking_errors) == 0 and syntax_passed and bool(diff.strip())

    return {
        "status": "ok",
        "ready_to_submit": can_submit,
        "files_modified_count": len(changed_files),
        "files_modified": [f[1] for f in changed_files if len(f) > 1],
        "lines_added": added_lines,
        "lines_deleted": deleted_lines,
        "syntax_verification_passed": syntax_passed,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "diff_preview": diff[:2500] if diff else ""
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-Submit Patch & Diff Auditor")
    parser.add_argument("--workspace", "-w", default=".", help="Path to workspace root")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    report = audit_workspace(ws)
    print(json.dumps(report, indent=2))
    if not report.get("ready_to_submit"):
        sys.exit(1)


if __name__ == "__main__":
    main()
