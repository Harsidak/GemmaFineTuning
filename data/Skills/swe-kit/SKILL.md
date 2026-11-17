---
name: swe-kit
description: Deterministic codebase locator (BM25 + AST), symbol scope reader, and pre-submit patch auditor + targeted pytest runner for /workspace.
---

# Skill: SWE Toolkit (`swe-kit`)

Pure-stdlib scripts running inside `/workspace` via `run_skill_script`. Outputs are compact JSON capped under 2,500 chars.

## Scripts

### 1. `scripts/locate.py` — Rank candidate source files, AST symbol spans & matching tests
```json
{"skill_name": "swe-kit", "file_path": "scripts/locate.py", "args": ["--query", "<issue title + key identifiers/messages>"]}
```
- Ranks package `.py` and `docs_src/` files using BM25/IDF + AST symbol definitions.
- Returns `top_candidates` (with exact `[start_line, end_line]` symbol bounds) and `recommended_tests`.

### 2. `scripts/read_symbol.py` — Read exact AST function/class or enclosing scope for a line
```json
{"skill_name": "swe-kit", "file_path": "scripts/read_symbol.py", "args": ["--file", "pkg/mod.py", "--symbol", "ClassName.method"]}
```
Or inspect around a line number / file outline:
- `["--file", "pkg/mod.py", "--line", "420"]`
- `["--file", "pkg/mod.py", "--outline"]`

### 3. `scripts/verify_and_audit.py` — Pre-submit syntax/leak/strip gatekeeper + targeted pytest
```json
{"skill_name": "swe-kit", "file_path": "scripts/verify_and_audit.py", "args": []}
```
Optional flags: `["--test-file", "tests/test_x.py", "-k", "keyword"]` or `["--no-tests"]`.
- Checks untracked scratch leaks in `/workspace` BEFORE `git add -N .`.
- Verifies `kept_source_files` is non-empty after stripping protected files (`tests/`, `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`).
- Runs `ast.parse` on every modified `.py` file and executes targeted `pytest -x -q`.
- Only call `submit_patch()` when `"ready_to_submit": true`.
