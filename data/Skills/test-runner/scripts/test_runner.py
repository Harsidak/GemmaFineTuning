#!/usr/bin/env python3
"""Targeted Test Runner Skill: Runs localized pytest suites without full-repo timeout.

Avoids the catastrophic failure mode of running bare `pytest` (which takes 15 minutes and times out).
Maps modified source files to their test modules and executes only the relevant test subset.

Usage:
    python test_runner.py --source fastapi/routing.py
    python test_runner.py --test-file tests/test_routing.py -k test_serialize
    python test_runner.py --changed-files
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_test_file_for_source(source_path: str, workspace: Path) -> list[str]:
    stem = Path(source_path).stem
    tests_dir = workspace / "tests"
    if not tests_dir.is_dir():
        return []

    candidates = []
    # Search tests/ for test_<stem>.py or <stem>_test.py
    for root, _, files in os.walk(tests_dir):
        for f in files:
            if f.endswith(".py") and (f == f"test_{stem}.py" or f == f"{stem}_test.py" or stem in f):
                rel = (Path(root) / f).relative_to(workspace).as_posix()
                candidates.append(rel)
    return candidates


def run_targeted_test(test_path: str, keyword: str = None, timeout: int = 90, workspace: Path = Path(".")) -> dict:
    cmd = [sys.executable, "-m", "pytest", test_path, "-v", "-q", "-p", "no:anyio", "-o", "timeout=0"]
    if keyword:
        cmd.extend(["-k", keyword])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace.resolve()) + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env
        )

        stdout = proc.stdout
        stderr = proc.stderr

        # Parse test metrics from pytest output
        passed = len(re.findall(r"PASSED", stdout))
        failed = len(re.findall(r"FAILED", stdout))
        errors = len(re.findall(r"ERROR", stdout))

        # Extract failure tracebacks if any
        failures_summary = []
        if failed or errors:
            fail_blocks = re.split(r"={3,}\s+FAILURES\s+={3,}", stdout)
            if len(fail_blocks) > 1:
                failures_summary = fail_blocks[1][:3000].strip()
            else:
                failures_summary = stdout[-2000:].strip()

        return {
            "status": "ok",
            "exit_code": proc.returncode,
            "verdict": "PASSED" if proc.returncode == 0 else "FAILED",
            "test_target": test_path,
            "keyword_filter": keyword,
            "tests_passed": passed,
            "tests_failed": failed,
            "tests_errors": errors,
            "failure_trace": failures_summary if failures_summary else None,
            "summary_line": stdout.splitlines()[-1] if stdout.splitlines() else ""
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "exit_code": -1,
            "verdict": "TIMEOUT",
            "message": f"Test run on {test_path} timed out after {timeout}s"
        }
    except Exception as exc:
        return {
            "status": "error",
            "exit_code": -1,
            "verdict": "ERROR",
            "message": str(exc)
        }


def main():
    parser = argparse.ArgumentParser(description="Targeted Unit Test Runner")
    parser.add_argument("--source", "-s", default=None, help="Modified source file to find tests for")
    parser.add_argument("--test-file", "-t", default=None, help="Exact test file path to run")
    parser.add_argument("-k", "--keyword", default=None, help="Pytest -k expression filter")
    parser.add_argument("--timeout", type=int, default=90, help="Max test execution timeout in seconds")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace root")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()

    test_targets = []
    if args.test_file:
        test_targets.append(args.test_file)
    elif args.source:
        test_targets = find_test_file_for_source(args.source, ws)
        if not test_targets:
            print(json.dumps({
                "status": "error",
                "message": f"No test files found in tests/ matching source {args.source}"
            }, indent=2))
            return

    if not test_targets:
        print(json.dumps({"status": "error", "message": "Must provide --source or --test-file"}))
        return

    # Run the primary target
    primary = test_targets[0]
    res = run_targeted_test(primary, keyword=args.keyword, timeout=args.timeout, workspace=ws)
    res["all_matching_test_files"] = test_targets
    print(json.dumps(res, indent=2))
    if res.get("verdict") != "PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()
