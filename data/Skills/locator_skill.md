# Skill: Code & Symbol Locator (`code_locator.py`)

## Objective
Quickly pinpoint the exact file, class, and function definitions responsible for a reported issue or feature request, without wasting turns or blowing the 32k context window on blind file exploration.

## Why this is necessary
- In SWE-bench benchmarks, only **12%** of problem statements mention the target file or module, and **36%** contain no obvious code anchor.
- The pre-computed graph tools often return test files (in FastAPI, 80% of graph nodes are test nodes).
- `code_locator.py` deterministically searches package source files (ignoring tests, docs, and build dirs), extracts camelCase and snake_case tokens, and ranks candidate source files and target tests.

## How to Invoke
```bash
python data/Skills/code_locator.py --query "<Problem statement or keywords>" --workspace /workspace
```

## Output Schema
Returns JSON with:
- `search_tokens`: Key identifiers extracted from the query.
- `top_candidates`: Ranked files with matched line numbers and AST symbol spans (`start_line`, `end_line`).
- `recommended_tests_to_run`: Candidate test files in `tests/` matching the affected modules.

## Example
```bash
python data/Skills/code_locator.py --query "serialize_response status_code" --workspace /workspace
```
Output:
```json
{
  "top_candidates": [
    {
      "file": "/workspace/fastapi/routing.py",
      "score": 45,
      "symbols": [{"name": "serialize_response", "line": 135, "end_line": 190}],
      "line_snippets": [{"line": 135, "text": "def serialize_response(..."}]
    }
  ],
  "recommended_tests_to_run": ["tests/test_routing.py"]
}
```
Use this output to immediately target `ast_reader.py` or `read_file` on lines 135–190.
