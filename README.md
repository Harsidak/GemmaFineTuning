# GemmaFineTuning

**An autonomous software engineering developer agent and post-training pipeline powered by Google's Gemma 4 model family for the [Kaggle Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview).**

`GemmaFineTuning` provides a unified, production-grade workspace to train, evaluate, and package autonomous coding agents under strict competition constraints: declarative ADK schemas, closed host registries, 1-GPU offline inference, and Pass@1 resolution benchmarks.

---

## ⚡ Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Sync the environment and install dependencies
uv sync

# 2. Check workspace readiness & status
uv run python pipeline/main.py --action status

# 3. Evaluate baseline zero-shot agent on validation tasks
uv run python pipeline/main.py --action evaluate --tasks data/val_tasks.jsonl

# 4. Package declarative submission bundle
uv run python pipeline/main.py --action package --model gemma-4-12b-it
```

---

## 📁 Repository Architecture

```text
GemmaFineTuning/
├── .agents/                              # Competition rules, scoring criteria & ADK constraints
│   └── rules.md                          # Official technical spec, tool schemas & winning playbook
├── resources/                            # Research knowledge base & paper index (clean in Git)
│   ├── INDEX.md                          # Literature catalogue (arXiv IDs, priorities & takeaways)
│   ├── initial_data_inspection.md        # Comprehensive dataset EDA & SWE-bench analysis
│   ├── _TEMPLATE.meta.yaml               # Sibling metadata schema for papers, blogs & repos
│   ├── papers/                           # Local PDF storage (gitignored)
│   ├── web/                              # Scraped reference guides & blogs (gitignored)
│   └── repos/                            # Cloned reference implementations (gitignored)
├── planning/                             # Competition roadmaps & failure post-mortems (gitignored)
│   ├── MISTAKES.md                       # Root Cause Analysis log & pre-flight checklist
│   └── ROADMAP.md                        # Milestone gates (baseline -> multi-agent -> scaling)
├── scripts/                              # Top-level engineering utilities & automation helpers
│   └── README.md                         # Guide for custom CLI and Kaggle automation scripts
├── kaggle/                               # Interactive launcher notebooks (run locally or on Kaggle)
│   ├── 00_master_pipeline_launcher.ipynb # Central controller (status, train, eval, package)
│   ├── 01_dataset_and_model_card.ipynb   # Dataset inspection & Gemma 4 Model Card launcher
│   ├── 02_agent_evaluation.ipynb         # Local simulation harness & Pass@1 evaluation launcher
│   └── 03_vllm_submission_packager.ipynb # vLLM offline serving & submission packaging launcher
├── pipeline/                             # Complete modular engine (uploadable as a Kaggle Dataset)
│   ├── main.py                           # Unified pipeline coordinator & CLI entrypoint
│   ├── configs/                          # Training & evaluation configuration files
│   │   ├── train_lora_sft.yaml           # SFT LoRA fine-tuning hyperparameters
│   │   ├── train_dpo_rl.yaml             # Trajectory preference optimization config
│   │   └── agent_defaults.yaml           # ADK declarative agent blueprint
│   ├── scripts/                          # Pipeline runner CLI entrypoints
│   │   ├── train.py                      # Training CLI runner (SFT / DPO)
│   │   ├── evaluate.py                   # Benchmark evaluator CLI runner
│   │   └── package_submission.py         # Automated submission.zip packager & validator
│   └── src/                              # Core engine implementation
│       ├── agent/                        # Declarative ADK builder, prompts & tool interfaces
│       ├── data/                         # Trajectory loaders, task samplers & processors
│       ├── eval/                         # Local sandbox evaluator & Pass@1 metric calculator
│       ├── models/                       # Gemma 4 QLoRA loaders & PEFT configs
│       ├── training/                     # SFT and DPO training loops
│       └── utils/                        # Packager validator & vLLM server helpers
├── experiments/                          # Strict empirical experiment tracking
│   ├── tracker.md                        # Master experiment leaderboard and run index
│   ├── template.yaml                     # Machine-readable experiment schema
│   ├── template.md                       # Human-readable experiment review (hypothesis, RCA, verdict)
│   └── exp_000_baseline/                 # Zero-shot baseline reference run
│       ├── config.yaml                   # Frozen baseline configuration
│       └── notes.md                      # Baseline execution review & Pass@1 result
├── data/                                 # Evaluation benchmark and training datasets
│   ├── val_tasks.jsonl                   # Held-out validation benchmark tasks (SWE-bench format)
│   └── train_trajectories.jsonl          # Fine-tuning trajectory samples
├── build/                                # Staging directories and compiled artifacts
│   ├── submission_staging/               # Scaffolding directory for declarative ADK files
│   └── gemma-4-developer-agent/          # Competition harness snapshots and sample submissions
├── submission.zip                        # Packaged competition submission bundle (gitignored)
├── CLAUDE.md                             # Agent operating spec & engineering rules
├── LICENSE                               # Apache License 2.0
├── pyproject.toml                        # Project dependencies and uv environment specification
├── uv.lock                               # Exact locked dependency tree (gitignored)
├── requirements.txt                      # Alternative pip-compatible dependency file
├── .gitignore                            # Clean Git exclusion rules
└── README.md                             # Master repository documentation
```

---

## 🎯 Competition Constraints & Principles

1. **Strictly Declarative (`adk-submission`)**:  
   Submissions must feature `agent.yaml` at the root of `submission.zip`. **No arbitrary Python entrypoints** (`agent.py`) are permitted. Custom multi-turn workflows operate via the **Agent-as-a-Tool** (`agent_tool`) sub-agent pattern.
2. **Model Consistency Rule**:  
   All root agents and sub-agents in a single package **must target the identical Gemma 4 base variant** (e.g. `gemma-4-31b-it-qat-w4a16-ct` or `gemma-4-12b-it`).
3. **Closed Tool Registries**:  
   Tools are restricted to host primitives: `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `code_graph_search`, and `submit_patch`.
4. **"Seeing Before Patching"**:  
   The primary failure mode of LLM agents is hallucinating edit locations. The agent utilizes dedicated sub-agents:
   * **`code_localizer`**: AST & graph search to identify exact fault lines before modifying code.
   * **`patch_engineer`**: Minimal diff synthesis avoiding unintended refactors.
   * **`test_verifier`**: Targeted unit test execution and regression checking.
5. **Offline & Budget Guardrails**:  
   Evaluation runs strictly offline on a single Kaggle GPU with a 9–12 hour wall-clock budget. All tests enforce strict timeouts to prevent command hangs.

---

## 🔬 Experimentation Workflow

To iterate rapidly and maintain scientific rigor:

1. **Plan & State Hypothesis**: Create a new experiment folder `experiments/exp_<ID>_<name>/`.
   * Fill out `hypothesis.md` with the mechanism, predicted Pass@1 delta, and explicit kill criteria before running code.
2. **Fine-Tune Adapter**:
   ```bash
   uv run python pipeline/main.py --action train --mode sft --config pipeline/configs/train_lora_sft.yaml
   ```
3. **Evaluate on Held-Out Tasks**:
   ```bash
   uv run python pipeline/main.py --action evaluate --tasks data/val_tasks.jsonl
   ```
4. **Log Root Cause Analysis & Verdict**:
   * Record failure modes and tool error breakdowns in `notes.md`.
   * Update the master leaderboard in [experiments/tracker.md](experiments/tracker.md).
   * Document any unexpected runtime hangs or crashes in [planning/MISTAKES.md](planning/MISTAKES.md).
5. **Package & Validate**:
   ```bash
   uv run python pipeline/main.py --action package --model gemma-4-12b-it
   ```

---

## 🚀 Kaggle Launcher Pattern

When submitting or iterating on Kaggle, upload the `pipeline/` directory as a Kaggle Dataset (e.g., `/kaggle/input/gemma-agent-pipeline/pipeline`). The launcher automatically detects the pipeline location and exposes the API:

```python
from pipeline.main import PipelineRunner

runner = PipelineRunner()
runner.status()                                    # Inspect environment & datasets
runner.evaluate(tasks_path="data/val_tasks.jsonl") # Test agent and calculate Pass@1
runner.package(model_name="gemma-4-12b-it")       # Generate validated submission.zip
```
