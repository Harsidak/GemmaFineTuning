# Skill: Resilient Syntax-Validated Editor (`safe_editor.py`)

## Objective
Apply surgical code modifications with automatic whitespace tolerance, ambiguity prevention, and immediate Python AST syntax validation.

## Why this is necessary
- Standard string replacement fails when line endings or indentation differ slightly from the model's memory.
- If a replacement introduces a syntax error (e.g. unclosed parenthesis, indentation error), standard editors commit the broken file.
- `safe_editor.py` tests exact match first, falls back to flexible whitespace matching, validates the modified file using `ast.parse()`, and automatically aborts without touching the disk if a `SyntaxError` is detected.

## How to Invoke

### 1. Perform a direct replacement:
```bash
python data/Skills/safe_editor.py --file fastapi/routing.py --old "def foo():" --new "def foo(x=None):"
```

### 2. Dry run check (verify match and AST validity without modifying file):
```bash
python data/Skills/safe_editor.py --file fastapi/routing.py --old "..." --new "..." --dry-run
```

### 3. Replace using multiline files (best for complex code blocks):
```bash
python data/Skills/safe_editor.py --file rich/console.py --old-file /tmp/old_block.py --new-file /tmp/new_block.py
```

## Output Schema
Returns JSON:
- `status`: `"ok"` or `"error"`.
- `strategy`: `"exact"` or `"flexible_whitespace"`.
- `diff`: Generated unified diff for verification.
- `syntax_error_line`: Reported if a syntax error was prevented.
