"""Experiment evaluation toolkit: arms as overlays, a concurrent task runner and a trace tally.

- `arms`   builds each experiment arm (a submission directory) from a base arm plus declared changes, checks it
           against the competition rules and zips it reproducibly.
- `runner` runs one arm's tasks through the official `swegemma` Evaluator with N tasks in flight.
- `tally`  reads the per-task rows and ATIF traces, classifies how each task ended and compares arms.

Pure Python + PyYAML: importable on a laptop without `swegemma`, `adk_submission` or a GPU. The Kaggle notebook
`kaggle/eval_exp016.ipynb` drives it; `python -m pipeline.exp_eval` exposes `build` and `tally` locally.
"""
