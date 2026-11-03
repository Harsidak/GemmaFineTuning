---
name: repro-runner
description: Run a small Python reproduction of the bug in an isolated temp dir (outside /workspace) and report pass/fail.
---

# Skill: Isolated Reproduction Test Runner (`repro_runner.py`)

## Objective
Safely replicate the bug reported in the problem statement using a minimal test script running in `/tmp`, verifying the failure before touching code and verifying the resolution after patching.

## Critical Harness Gotchas Prevented
- **Git Pollution Trap**: `submit_patch()` runs `git add -N . && git diff HEAD` in `/workspace`. If an agent creates `/workspace/repro.py`, that file is permanently captured into the git diff submitted for evaluation!
- `repro_runner.py` executes code strictly in `/tmp/repro_test.py` with `/workspace` on `PYTHONPATH`, completely eliminating git diff pollution.
- Running a reproduction test *before* making edits provides an infallible ground-truth check (verifying that the failure is reproducible).

## How to Invoke

### 1. Run inline python reproduction code:
```bash
python scripts/repro_runner.py --code "from fastapi import FastAPI; app = FastAPI(); assert app is not None"
```

### 2. Run with pytest runner:
```bash
python scripts/repro_runner.py --code "def test_issue(): assert False" --pytest
```

### 3. Run from an existing scratch file:
```bash
python scripts/repro_runner.py --file /tmp/repro_script.py
```

## Workflow
1. **Pre-Fix**: Write a 5-line repro test that fails (verdict `FAILED`, exit code `1`).
2. **Apply Patch**: Modify the source code using `safe_editor.py` or `edit_file`.
3. **Post-Fix**: Re-run `repro_runner.py`. It must now return verdict `PASSED` (exit code `0`).

## Running it
Preferred: `run_skill_script` with skill_name `repro-runner` and file_path `scripts/repro_runner.py`, passing the arguments below.
If that tool is unavailable or fails, use `load_skill_resource` to read `scripts/repro_runner.py`, save it to `/tmp/repro_runner.py`
(never inside /workspace), then `run_command`: `python /tmp/repro_runner.py <arguments>`.
