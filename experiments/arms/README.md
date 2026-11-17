# Experiment arms

An **arm** is one submission variant an experiment compares. Each arm is a folder with an `arm.yaml`:

* **Full arm** (`base: null`): the whole submission sits in `submission/`. Only `ctrl_022` is one: the 0.22
  leaderboard zip, unpacked byte for byte.
* **Overlay arm** (`base: <arm id>`): the base arm plus the changes it declares, and nothing else.

```yaml
id: exp016_a1_skill_first
experiment: EXP-016
base: exp016_a0_pruned
description: what this arm tests, in one or two sentences
changes:
  files:        # submission path -> repo-relative source file or directory (copied, __pycache__/*.pyc dropped)
    prompts/system.md: pipeline/submission_template/prompts/system.md
  yaml_set:     # submission YAML file -> {dotted.key: value}; a null value deletes the key
    eval_config.yaml: {evaluation.max_tool_calls: 30}
  remove: []    # submission paths to delete
```

`python -m pipeline.exp_eval build <arm>` writes `submission/`, a reproducible `submission.zip` and an
`arm_manifest.json`. The manifest lists the files that differ from the base arm (`changed_vs_base`) and the
per-task budget the scorer will apply.

The build fails if:

* a declared change has no effect;
* the result breaks a competition rule: the base model, tool names, file types, `.py` outside `skills/`, generation
  limits, eval_config keys, missing `!include`s or a missing adapter directory.

The build warns when the prompt states a different call or minute budget than `eval_config.yaml`.

## Current arms

| Arm | Base | What changes | Budget |
|---|---|---|---|
| `ctrl_022` | — | The 0.22 submission (9 tools, 3 of them dead; no skills; no adapter) | 28 calls / 8 min |
| `exp016_a0_pruned` | `ctrl_022` | Drops the 3 dead graph tools and the prompt line banning them | 28 / 8 |
| `exp016_a1_skill_first` | `exp016_a0_pruned` | Adds the `swe-kit` skill, the skill-first prompt and the 30-call budget that prompt states | 30 / 8 |

## Adding an arm

1. Make a folder named after the arm with an `arm.yaml`. Put new prompt files under `<arm>/files/`.
2. Declare one variable per arm. If a change needs two edits, for example a prompt and its budget, say so in
   `description`.
3. Run `uv run python -m pipeline.exp_eval build <arm>` and read the `changed vs` line.
4. Run `uv run python -m pytest pipeline/exp_eval`.
5. Commit and push, update the repo dataset, then run
   `uv run scripts/kaggle_push.py -exp016_eval --set arms=ctrl_022,<arm>`.

The control is always `ctrl_022` until a winner is adopted. Re-baseline then (tracker.md: CTRL-31B).
