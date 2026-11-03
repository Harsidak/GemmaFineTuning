# Skill: Targeted Test Runner (`test_runner.py`)

## Objective
Execute localized, targeted test suites against modified repository code without triggering the global 300-second command timeout or running external network-dependent tests.

## Why this is necessary
- Running bare `pytest` in repositories like FastAPI or Requests takes 10–20 minutes and crashes on offline network checks, exceeding the harness's 300s single-command limit.
- `test_runner.py` inspects the modified source file, identifies its matching unit test file under `tests/`, and executes only that specific test file with `-p no:anyio -o timeout=0`.

## How to Invoke

### 1. Auto-discover and run tests for a modified source file:
```bash
python data/Skills/test_runner.py --source fastapi/routing.py
```

### 2. Run a specific test file with a keyword filter:
```bash
python data/Skills/test_runner.py --test-file tests/test_routing.py -k test_serialize_response
```

### 3. Specify custom timeout:
```bash
python data/Skills/test_runner.py --source rich/console.py --timeout 60
```

## Output Schema
Returns JSON:
- `verdict`: `"PASSED"` or `"FAILED"`.
- `tests_passed`, `tests_failed`, `tests_errors`.
- `failure_trace`: Extracted pytest failure traceback (truncated to 3,000 chars) for immediate bug diagnosis.
