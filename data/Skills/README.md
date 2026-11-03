# GemmaFineTuning Developer Skills Library

This directory contains the production-grade, deterministic Python skills and prompt specifications designed to solve the 129 public SWE-bench tasks and the 389 mined verification tasks in `DataMining/final_output/`.

---

## 📂 Directory Structure

```text
data/skills/
├── SystemPrompt.md            # master system prompt
├── README.md                  # this file
├── code-locator/   SKILL.md + scripts/code_locator.py    (shipped by default)
├── repro-runner/   SKILL.md + scripts/repro_runner.py    (shipped by default)
├── test-runner/    SKILL.md + scripts/test_runner.py     (shipped by default)
├── patch-auditor/  SKILL.md + scripts/patch_auditor.py   (shipped by default)
├── ast-reader/     SKILL.md + scripts/ast_reader.py      (opt-in)
└── safe-editor/    SKILL.md + scripts/safe_editor.py     (opt-in; overlaps edit_file)
```

The harness only loads skills as folders containing `SKILL.md` (front-matter `name` = folder name, plus `description`),
listed under `skills:` in agent.yaml. `ADKSubmissionBuilder` copies them to `skills/` in the submission.

---

## 🎯 Benchmark Alignment Matrix

| Empirical Finding from 129 Tasks & 389 Mined Tasks | Critical Failure Mode | Skill Solution |
| :--- | :--- | :--- |
| **88% of issues do not name the file** | Blindly browsing files from line 1 wastes turn and token budget | `code_locator.py` tokenizes problem statement and ranks package files by frequency & AST definitions |
| **Source files are 1,000–4,500 lines long (`read_file` capped at 150 lines)** | Paging through whole files overflows context window (32k tokens) | `ast_reader.py` extracts exact line spans (`start_line`, `end_line`) for any target function/class |
| **Scratch scripts leak into submitted patch** | Creating `/workspace/repro.py` pollutes `git diff HEAD` and causes submission failure | `repro_runner.py` executes test scripts strictly in `/tmp/` with zero git pollution |
| **`edit_file` whitespace and syntax errors** | Slight indentation mismatches or syntax errors fail the task | `safe_editor.py` provides whitespace-flexible matching and validates with `ast.parse` before writing |
| **Full `pytest` times out after 300s** | Running bare `pytest` takes 15 minutes or hangs on offline network checks | `test_runner.py` maps the modified file to its matching unit test file in `tests/` |
| **`submit_patch()` ends session immediately** | Submitting a broken, empty, or tampered diff gives 0.0 with no second chance | `patch_auditor.py` audits syntax, diff size, protected files, and untracked files before submission |
