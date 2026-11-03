---
name: patch-auditor
description: Check the pending diff before submit_patch: syntax, stray scratch files, edits to pytest.ini/conftest.py.
---

# Skill: Pre-Submit Patch & Diff Auditor (`patch_auditor.py`)

## Objective
Act as an automated final quality gate before invoking `submit_patch()`, catching all silent failure conditions, syntax errors, and protected-file tampering.

## Why this is necessary
- In `swegemma`, calling `submit_patch()` **immediately terminates** the agent session. If your patch has a syntax error, is empty, or modifies protected files, you score 0.0 with no second chance!
- The competition harness forcefully strips and discards any changes to `tests/`, `conftest.py`, `pytest.ini`, or `pyproject.toml`.
- Untracked files in `/workspace` will be included in the diff via `git add -N .`.
- `patch_auditor.py` audits `git diff HEAD`, parses all modified Python files with `ast.parse()`, verifies no protected files were altered, flags untracked files, and gives an unambiguous `ready_to_submit: true/false` verdict.

## How to Invoke
```bash
python scripts/patch_auditor.py --workspace /workspace
```

## Output Schema
Returns JSON:
```json
{
  "status": "ok",
  "ready_to_submit": true,
  "files_modified_count": 1,
  "files_modified": ["fastapi/routing.py"],
  "lines_added": 8,
  "lines_deleted": 2,
  "syntax_verification_passed": true,
  "blocking_errors": [],
  "warnings": [],
  "diff_preview": "--- a/fastapi/routing.py\n..."
}
```

## Rule of Thumb
**Never call `submit_patch()` until `patch_auditor.py` returns `"ready_to_submit": true`!**

## Running it
Preferred: `run_skill_script` with skill_name `patch-auditor` and file_path `scripts/patch_auditor.py`, passing the arguments below.
If that tool is unavailable or fails, use `load_skill_resource` to read `scripts/patch_auditor.py`, save it to `/tmp/patch_auditor.py`
(never inside /workspace), then `run_command`: `python /tmp/patch_auditor.py <arguments>`.
