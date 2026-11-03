---
name: ast-reader
description: Print the exact line span and source of a class or function in a file.
---

# Skill: AST Code Scope Inspector (`ast_reader.py`)

## Objective
Inspect large Python files (1,000–4,500 lines) with high precision, extracting exact function definitions and enclosing scopes without violating the 150-line `read_file` truncation ceiling.

## Why this is necessary
- `read_file` in the competition harness caps output to **150 lines** and **10,000 characters**.
- High-frequency files (e.g. `fastapi/routing.py`, `rich/console.py`) are multi-thousand line modules. Blindly reading line 1 to 150 wastes turns and budget.
- `ast_reader.py` parses the Python AST to give you the exact line boundaries of any function or class, or reveals which function encloses a specific line number.

## How to Invoke

### 1. Inspect a specific symbol:
```bash
python scripts/ast_reader.py --file fastapi/routing.py --symbol serialize_response
```

### 2. Inspect scope around a specific line:
```bash
python scripts/ast_reader.py --file rich/console.py --line 512 --context 20
```

### 3. List all functions and classes in a module with line ranges:
```bash
python scripts/ast_reader.py --file fastapi/applications.py --list-symbols
```

## Output Schema
Returns JSON with:
- `start_line` / `end_line`: Exact 1-indexed line numbers.
- `code`: Line-numbered source code for the requested symbol.
- `enclosing_symbol`: The parent function or class if searching by line number.

## Running it
Preferred: `run_skill_script` with skill_name `ast-reader` and file_path `scripts/ast_reader.py`, passing the arguments below.
If that tool is unavailable or fails, use `load_skill_resource` to read `scripts/ast_reader.py`, save it to `/tmp/ast_reader.py`
(never inside /workspace), then `run_command`: `python /tmp/ast_reader.py <arguments>`.
