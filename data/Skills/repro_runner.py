#!/usr/bin/env python3
"""Reproduction Test Runner Skill: Safely reproduces and verifies fixes without polluting git diff.

HARNESS RULE: Scratch scripts in /workspace appear in `git diff` unless cleaned up!
This skill writes isolated reproduction tests to /tmp/repro_test.py, executes them in the
sandbox python environment, and verifies whether a bug reproduces (fails) and is fixed (passes).

Usage:
    python repro_runner.py --code "import fastapi; assert fastapi.__version__"
    python repro_runner.py --test-file /path/to/test.py
    python repro_runner.py --code "def test_issue(): assert False" --pytest
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_repro(code: str, use_pytest: bool = False, timeout: int = 60) -> dict:
    """Executes code in /tmp and returns execution verdict."""
    tmp_dir = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    script_path = tmp_dir / f"_swegemma_repro_{os.getpid()}.py"

    try:
        script_path.write_text(code, encoding="utf-8")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "/workspace" + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")

        if use_pytest:
            cmd = [sys.executable, "-m", "pytest", str(script_path), "-v", "-q"]
        else:
            cmd = [sys.executable, str(script_path)]

        proc = subprocess.run(
            cmd,
            cwd="/workspace" if Path("/workspace").is_dir() else ".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env
        )

        return {
            "status": "ok",
            "exit_code": proc.returncode,
            "verdict": "PASSED" if proc.returncode == 0 else "FAILED",
            "stdout": proc.stdout[:3000],
            "stderr": proc.stderr[:3000],
            "script_executed": str(script_path)
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "exit_code": -1,
            "verdict": "TIMEOUT",
            "message": f"Execution timed out after {timeout} seconds"
        }
    except Exception as exc:
        return {
            "status": "error",
            "exit_code": -1,
            "verdict": "ERROR",
            "message": str(exc)
        }
    finally:
        if script_path.exists():
            try:
                script_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Isolated Repro Test Runner")
    parser.add_argument("--code", "-c", default=None, help="Python code snippet to execute in /tmp")
    parser.add_argument("--file", "-f", default=None, help="Existing file containing reproduction script")
    parser.add_argument("--pytest", action="store_true", help="Run with pytest instead of python directly")
    parser.add_argument("--timeout", type=int, default=45, help="Execution timeout in seconds")
    args = parser.parse_args()

    code = args.code
    if not code and args.file:
        p = Path(args.file)
        if p.is_file():
            code = p.read_text(encoding="utf-8")
        else:
            print(json.dumps({"status": "error", "message": f"File not found: {args.file}"}))
            sys.exit(1)

    if not code:
        print(json.dumps({"status": "error", "message": "Must provide either --code or --file"}))
        sys.exit(1)

    result = run_repro(code, use_pytest=args.pytest, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
