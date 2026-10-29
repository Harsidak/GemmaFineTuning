# Deep Forensic Data Inspection & Architectural Research Report

> **Competition:** [Google - The Gemma 4 Developer Agent Competition (Kaggle)](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)  
> **Author:** Research & Engineering Team  
> **Status:** Active Reference Document  
> **Target Score:** 0.80+ Pass@1 on Private Test Set  

---

## Table of Contents

1. [Executive Summary & Core Paradigm](#1-executive-summary--core-paradigm)
2. [Forensic Analysis of `tasks.jsonl` (The 129 Benchmark Tasks)](#2-forensic-analysis-of-tasksjsonl-the-129-benchmark-tasks)
   - [2.1. Repository Breakdown & Archetypes](#21-repository-breakdown--archetypes)
   - [2.2. Problem Statement Distribution](#22-problem-statement-distribution)
   - [2.3. Ground-Truth Patch Geometry & Line Percentiles](#23-ground-truth-patch-geometry--line-percentiles)
   - [2.4. Target Directories Touched](#24-target-directories-touched)
   - [2.5. Problem Categorization & Keyword Typology](#25-problem-categorization--keyword-typology)
3. [Pre-Computed Graph & Vector Intelligence Layer](#3-pre-computed-graph--vector-intelligence-layer)
   - [3.1. AST Call Graph Schema (`graphs/*.json`)](#31-ast-call-graph-schema-graphsjson)
   - [3.2. 256-Dimensional Vector Embeddings (`embeddings/*.npz`)](#32-256-dimensional-vector-embeddings-embeddingsnpz)
   - [3.3. Exact 1-to-1 Node-to-Vector Correspondence](#33-exact-1-to-1-node-to-vector-correspondence)
4. [The `swegemma` Evaluation Harness Architecture](#4-the-swegemma-evaluation-harness-architecture)
   - [4.1. Cooperating Library Stack](#41-cooperating-library-stack)
   - [4.2. Compute Specification: 4x NVIDIA L4 GPUs (96 GB VRAM)](#42-compute-specification-4x-nvidia-l4-gpus-96-gb-vram)
   - [4.3. Single Base Model Rule & Allowed Models](#43-single-base-model-rule--allowed-models)
   - [4.4. Submission Storage & Context Ceilings](#44-submission-storage--context-ceilings)
   - [4.5. Two-Phase Hermetic Sandbox Lifecycle](#45-two-phase-hermetic-sandbox-lifecycle)
5. [The 9 Built-In Tools Reference (`swegemma.tools`)](#5-the-9-built-in-tools-reference-swegemmatools)
   - [5.1. Execution & Lifecycle Tools (`run_command`, `submit_patch`, `get_status`)](#51-execution--lifecycle-tools-run_command-submit_patch-get_status)
   - [5.2. Workspace File Tools (`read_file`, `edit_file`, `write_file`)](#52-workspace-file-tools-read_file-edit_file-write_file)
   - [5.3. Graph & Semantic Tools (`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`)](#53-graph--semantic-tools-search_similar_code-get_code_neighbors-get_code_subgraph)
6. [Official Sample Submission Architecture (Multi-LoRA & Prompts)](#6-official-sample-submission-architecture-multi-lora--prompts)
   - [6.1. Declarative Routing in `agent.yaml`](#61-declarative-routing-in-agentyaml)
   - [6.2. Delegated Analysis in `sub_agents/code_analyzer.yaml`](#62-delegated-analysis-in-sub_agentscode_analyzeryaml)
   - [6.3. Inference & Thinking Config in `configs/sampling.yaml`](#63-inference--thinking-config-in-configssamplingyaml)
   - [6.4. Official System Prompt Directives (`prompts/system.md`)](#64-official-system-prompt-directives-promptssystemmd)
7. [The 4 Fatal Anti-Patterns & Instant Disqualification Risks](#7-the-4-fatal-anti-patterns--instant-disqualification-risks)
8. [The Winning Master Strategy: Scaling LoRA SFT to 0.82+ Pass@1](#8-the-winning-master-strategy-scaling-lora-sft-to-082-pass1)
   - [8.1. Why Reinforcement Learning is Sub-Optimal for this Task](#81-why-reinforcement-learning-is-sub-optimal-for-this-task)
   - [8.2. Phase 1: Mining the 129 Public Tasks into Gold Trajectories](#82-phase-1-mining-the-129-public-tasks-into-gold-trajectories)
   - [8.3. Phase 2: Archetype-Targeted GitHub PR Mining](#83-phase-2-archetype-targeted-github-pr-mining)
   - [8.4. Phase 3: Teacher-Forced Synthetic Trajectory Synthesis](#84-phase-3-teacher-forced-synthetic-trajectory-synthesis)
   - [8.5. Multi-LoRA Dual-Specialization Architecture](#85-multi-lora-dual-specialization-architecture)

---

## 1. Executive Summary & Core Paradigm

The **Google - The Gemma 4 Developer Agent Competition** represents a structural departure from traditional Kaggle challenges. Rather than predicting flat tables or generating unverified text, competitors are evaluated on **autonomous agentic software engineering**.

Submitted agents operate inside an isolated Docker sandbox running **Python 3.13**, equipped with pre-installed dependencies via offline wheels. The agent must independently inspect issue descriptions, localize root causes in complex codebases using pre-computed Abstract Syntax Tree (AST) graphs and dense vector embeddings, synthesize minimal unified git diffs, verify fixes against unit test suites, and produce a patch that passes private verification tests.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CORE COMPETITION METRICS                         │
├─────────────────────────────────────────────────────────────────────────┤
│ • Benchmark Metric:              Pass@1 Functional Correctness Rate     │
│ • Benchmark Size:                129 Public Tasks / ~120 Private Tasks  │
│ • Hardware Allowance:            4x NVIDIA L4 GPUs (96 GB Total VRAM)   │
│ • Execution Paradigm:            Strict Declarative ADK (agent.yaml)    │
│ • Code Entrypoint Allowed:       NONE (Arbitrary agent.py is banned)    │
│ • Maximum Target Turns:          < 8–10 turns per task                  │
│ • Submission Unpacked Cap:       < 3 GiB (including all LoRA weights)   │
│ • Output Submission Format:      submission.parquet (id, prediction)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Forensic Analysis of `tasks.jsonl` (The 129 Benchmark Tasks)

The public development dataset consists of **129 benchmark tasks** stored in `build/gemma-4-developer-agent/tasks.jsonl`. Each entry represents a real-world software issue paired with its base repository commit, ground-truth reference fix (`patch`), and verification tests (`test_patch`).

### 2.1. Repository Breakdown & Archetypes

Empirical analysis of `tasks.jsonl` yields the following exact distribution:

```text
Repository Breakdown across 129 Public Tasks:
├── fastapi/fastapi:    67 tasks  (51.9%)
├── Textualize/rich:    48 tasks  (37.2%)
├── psf/requests:       13 tasks  (10.1%)
└── encode/httpx:        1 tasks  (0.8%)
```

```
┌────────────────────────────────────────────────────────────────────────┐
│               REPOSITORY ARCHETYPE DISTRIBUTION                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   FastAPI (51.9%)            Rich (37.2%)        Requests/HTTPX (10.9%)│
│  █████████████████████     ███████████████       ████                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **FastAPI + Rich constitute 89.1% of the public dataset.**  
> While the private test set is drawn from closed-source internal repositories, it was generated using the **exact same curation pipeline**. Both FastAPI and Rich represent modern, pure Python libraries heavily reliant on `typing`, Pydantic models, ASGI/async protocols, and ANSI layout algorithms.

---

### 2.2. Problem Statement Distribution

The `problem_statement` represents the natural language bug report or pull request description passed directly to the agent in the initial user turn:

| Metric | Characters | Word Count |
| :--- | :---: | :---: |
| **Minimum** | 22 chars | 4 words |
| **25th Percentile** | 165 chars | 28 words |
| **Median (50th Percentile)** | **342 chars** | **58 words** |
| **Mean** | 684 chars | 106 words |
| **75th Percentile** | 820 chars | 134 words |
| **Maximum** | 4,812 chars | 772 words |

* **Observation:** Most problem statements are concise (under 60 words). They often contain explicit symbol names (e.g. `_prepare_response_content`), error types (e.g. `RequestValidationError`), or sample reproduction code snippets.

---

### 2.3. Ground-Truth Patch Geometry & Line Percentiles

Analyzing the unified git diffs in `patch` across all 129 tasks reveals a profound structural reality: **the fixes are surgical and small.**

```text
Number of Files Modified per Ground-Truth Patch:
├── 1 file modified:    91 tasks  (70.5%)
├── 2 files modified:   19 tasks  (14.7%)
├── 3 files modified:    5 tasks  (3.9%)
├── 4 files modified:    3 tasks  (2.3%)
└── 5+ files modified:  11 tasks  (8.6%)
```

* **85.2% of all tasks touch $\le 2$ files.**
* **70.5% touch exactly ONE file.**

#### Diff Line Percentiles (Ground-Truth Solutions):
* **25th Percentile:** `13 diff lines` (approx. 2 to 4 lines of logic changed)
* **Median (50th Percentile):** `32 diff lines` (approx. 5 to 10 lines of logic changed)
* **75th Percentile:** `85 diff lines`
* **90th Percentile:** `214 diff lines`

> [!TIP]
> **Winning Implication:** The agent must be trained to produce **minimal, surgical edits** using `edit_file`. Massive rewrites or refactoring of entire modules is an immediate failure pattern.

---

### 2.4. Target Directories Touched

Analysis of `--- a/<filepath>` in ground-truth patches demonstrates the exact subdirectories modified:

```text
Top Directories Modified in Solutions:
├── fastapi/   : 129 files
├── rich/      : 91 files
├── docs_src/  : 34 files  (Executable documentation examples in FastAPI)
├── src/       : 22 files  (psf/requests and encode/httpx)
├── scripts/   : 2 files
└── tests/     : 0 files   (NEVER TOUCHED!)
```

* **Target Directories in Verification `test_patch`:**
  * `tests/`: 282 files (100% of all verification tests reside under `tests/`).
  * The harness strictly grades fixes by running tests from `tests/` against the agent's changes in source modules.

---

### 2.5. Problem Categorization & Keyword Typology

Analyzing problem statements reveals distinct recurring problem classes:

```text
Top Keywords in Problem Statements:
├── 'fix':         95 tasks (73.6%)
├── 'add':         61 tasks (47.3%)
├── 'test':        47 tasks (36.4%)
├── 'type':        40 tasks (31.0%)  <-- Heavy typing / validation focus
├── 'update':      34 tasks (26.4%)
├── 'doc':         33 tasks (25.6%)  <-- Documentation code fixes (docs_src/)
├── 'support':     31 tasks (24.0%)
├── 'error':       26 tasks (20.2%)
├── 'bug':         25 tasks (19.4%)
├── 'allow':       14 tasks (10.9%)
├── 'async':       12 tasks (9.3%)
└── 'validation':   9 tasks (7.0%)
```

The bugs cluster into four dominant categories:
1. **Validation & Type Casting:** Handling union types, optional fields, custom Pydantic validators, or unexpected `None`/empty payloads.
2. **Protocol & Stream Edge Cases:** HTTP redirect loops, header casing, chunked transfer encoding, and socket cleanup.
3. **Rendering & Layout Edge Cases:** Rich console ANSI code parsing, multi-byte character wrapping, table padding calculations.
4. **Docs Executable Code Alignment:** Ensuring tutorial code samples under `docs_src/` conform to newer API versions.

---

## 3. Pre-Computed Graph & Vector Intelligence Layer

Rather than forcing the LLM to run slow, broad disk searches (`grep`, `find`), the dataset provides **pre-computed Abstract Syntax Tree (AST) call graphs** and **dense vector embeddings** for every task snapshot.

```
                    PRE-COMPUTED CODE INTELLIGENCE
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   graphs/*.json                                   embeddings/*.npz
(NetworkX Call Graphs)                         (256-dim Dense Vectors)
  • 4,200+ Nodes per repo                        • 4,200+ Vectors per repo
  • Symbols, methods, classes                    • Cosine similarity search
  • 'calls', 'imports' edges                     • float32 embedding arrays
         │                                                 │
         ▼                                                 ▼
   get_code_neighbors()                            search_similar_code()
   get_code_subgraph()
```

### 3.1. AST Call Graph Schema (`graphs/*.json`)

There are **127 serialized NetworkX graph files** named `<repo_short>_<base_commit>.json` (e.g. `fastapi_016ab760faba1813ba235d5582a014a58bd2f417.json`).

* **Graph Metadata:**
  * `directed: true` (directed multigraph)
  * `multigraph: true` (permits multiple call/reference edges between symbol pairs)
* **Node Schema:**
  ```json
  {
    "id": "fastapi.routing._prepare_response_content",
    "name": "fastapi.routing._prepare_response_content",
    "text": "def _prepare_response_content(\n    res: Any,\n    *, ...\n): ...\n"
  }
  ```
  * `id`: Dotted symbol path uniquely identifying the Python class or function.
  * `text`: **The complete Python source code definition** of that function/class at `base_commit`.
* **Edge Schema:**
  ```json
  {
    "source": "fastapi.routing.serialize_response",
    "target": "fastapi.routing._prepare_response_content",
    "type": "calls",
    "key": 0
  }
  ```
  * `type`: Structural relationship (`"calls"`, `"imports"`, or `"references"`).

---

### 3.2. 256-Dimensional Vector Embeddings (`embeddings/*.npz`)

Alongside each graph JSON is a matching `.npz` archive (e.g. `fastapi_016ab760faba1813ba235d5582a014a58bd2f417.npz`).
* Each `.npz` archive contains a dictionary mapping node IDs to `float32` arrays of shape `(256,)`.
* Used by the tool `search_similar_code(query: str, k: int = 10)` to compute cosine similarity against all node vectors in memory in sub-second time.

---

### 3.3. Exact 1-to-1 Node-to-Vector Correspondence

We performed an automated intersection check between the graph JSON and embedding NPZ for `fastapi_016ab...`:
* Total Graph Nodes: **4,287**
* Total Embedding Keys: **4,287**
* Intersection: **4,287 (100% exact match)**

Every code symbol in the repository AST is pre-embedded and indexed. The agent can jump directly from a natural language symptom to candidate symbol definitions in **a single tool call**.

---

## 4. The `swegemma` Evaluation Harness Architecture

The competition grading infrastructure is detailed in `HARNESS_README.md`. It is built on three tightly coupled Python packages:

### 4.1. Cooperating Library Stack

1. **`adk-submission`**: Declarative agent compiler and inference server manager. Replaces standard ADK dynamic imports with a sandboxed compiler (`compile_submission`). Traverses `agent.yaml`, resolves references against sealed host registries (`ToolRegistry`, `ModelRegistry`), and boots local inference servers.
2. **`adk-eval-core`**: Sandbox execution backend (`DockerContainer`), resilient 3-tier file editing engine (`apply_replacement`), and token/cost budgeting.
3. **`swegemma`**: The SWE-bench evaluation engine. Binds the 9 tools, manages the two-container lifecycle, and executes hermetic Pytest scoring.

---

### 4.2. Compute Specification: 4x NVIDIA L4 GPUs (96 GB VRAM)

Evaluation runs on a host machine equipped with **4 × NVIDIA L4 GPUs**:
* **Architecture:** NVIDIA Ada Lovelace, 24 GB GDDR6 per GPU (`96 GB` total VRAM).
* **vLLM Serving Configuration:**
  * `tensor_parallel_size = 4` (sharded across all 4 GPUs).
  * `gpu_memory_utilization = 0.90` (`~86.4 GB` usable VRAM).
  * `max_model_len = 32768` tokens context ceiling.
  * Native tool-call and reasoning parsing: `tool_call_parser = "gemma4"`, `reasoning_parser = "gemma4"`.
  * Dynamic LoRA right-sizing: `enable_lora = True`, `max_lora_rank <= 128`.

---

### 4.3. Single Base Model Rule & Allowed Models

The competition enforces a **strict single base model rule** (`ALLOWED_MODEL_NAMES`):

| Permitted Base Model | Weight Precision | Footprint on 4x L4 | Competition Status |
| :--- | :---: | :---: | :---: |
| **`gemma-4-31b-it-qat-w4a16-ct`** | W4A16 (INT4 Quantized) | `~16–18 GB` (`~4.5 GB / GPU`) | **Permitted & Recommended** |
| **`gemma-4-12b-it-qat-w4a16-ct`** | W4A16 (INT4 Quantized) | `~6–8 GB` (`~1.8 GB / GPU`) | **Permitted** |
| **`gemma-4-e4b-it-qat-w4a16-ct`** | W4A16 (INT4 Quantized) | `~3–4 GB` (`~1.0 GB / GPU`) | **Permitted** |
| **`gemma-4-e2b-it-qat-w4a16-ct`** | W4A16 (INT4 Quantized) | `~2–3 GB` (`~0.7 GB / GPU`) | **Permitted** |

> [!IMPORTANT]
> **Single Model Rule:** Every agent, sub-agent, and tool in `submission.zip` **must declare the identical base model** (e.g. all `gemma-4-31b-it-qat-w4a16-ct`). Mixing models causes immediate validation rejection.

---

### 4.4. Submission Storage & Context Ceilings

* **Primary Archive Limit:** Total unpacked submission size must be **`< 3 GiB`** (`3,221,225,472` bytes), including all LoRA weights in `adapters/`.
* **Context Ceiling:** Total input prompt + thinking tokens + output tokens cannot exceed **32,768 tokens**.
* **Container Limits:** Each sandbox container gets **4 GiB RAM** and **2 vCPUs**.

---

### 4.5. Two-Phase Hermetic Sandbox Lifecycle

Every task is evaluated across two isolated Docker containers:

```text
EVALUATION LIFECYCLE (Two Containers)

Phase 1: Agent Execution (Container A)
┌────────────────────────────────────────────────────────┐
│ 1. Extract repo snapshot at base_commit                │
│ 2. Install wheels via /wheels/ (no internet)           │
│ 3. Create clean baseline git commit in HEAD            │
│ 4. Agent runs tools (search, read, edit, test)         │
│ 5. Agent calls submit_patch()                          │
│ 6. Extract unified diff: git add -N . && git diff HEAD │
└──────────────────────────┬─────────────────────────────┘
                           │ agent_patch
                           ▼
Phase 2: Verification (Container B)
┌────────────────────────────────────────────────────────┐
│ 1. Fresh container with clean repo snapshot            │
│ 2. 4-pass resilient git apply of agent_patch           │
│ 3. Reset tests: git checkout HEAD tests/               │
│ 4. Apply ground-truth task.test_patch                  │
│ 5. Run pytest on modified test files                   │
│ 6. exit_code == 0 & JUnit XML valid -> PASS (1.0)      │
└────────────────────────────────────────────────────────┘
```

---

## 5. The 9 Built-In Tools Reference (`swegemma.tools`)

The harness provides 9 tools. Every tool returns a structured JSON string:
* Success: `{"status": "ok", ...}`
* Error: `{"status": "error", "error_type": "...", "error_message": "...", "details": {...}}`

---

### 5.1. Execution & Lifecycle Tools (`swegemma/tools/execution.py`)

#### 1. `run_command(command: str) -> str`
* Executes a bash command inside `/workspace`.
* **Budget Gated:** Counts toward tool call limit.
* **Timeout:** 300 seconds default.
* **Output Truncation:** `stdout` and `stderr` are capped at **5,000 characters**.
* *Return Schema:*
  ```json
  {"status": "ok", "stdout": "...", "stderr": "...", "exit_code": 0}
  ```

#### 2. `submit_patch() -> str`
* Captures `git diff --binary HEAD` from `/workspace`, sets `patch_submitted = True`, and terminates the agent loop on turn completion.
* **Budget Gated:** **NO** (`count_tool_call=False`). Can be called even when tool calls are exhausted.
* *Return Schema:*
  ```json
  {"status": "ok", "patch_size": 1420, "files_changed": 2}
  ```

#### 3. `get_status() -> str`
* Returns remaining tool calls, remaining time, and patch submission status.
* **Budget Gated:** **NO** (`count_tool_call=False`).
* *Return Schema:*
  ```json
  {
    "status": "ok",
    "tool_calls_used": 6,
    "tool_calls_remaining": 44,
    "max_tool_calls": 50,
    "time_seconds_remaining": 3200.0,
    "patch_submitted": false
  }
  ```

---

### 5.2. Workspace File Tools (`swegemma/tools/workspace.py`)

#### 4. `read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str`
* Reads file content with 1-indexed line slicing.
* **Budget Gated:** Yes.
* **Dual Truncation Cap:** Output is capped at **150 lines** AND **10,000 characters**.
* *Return Schema:*
  ```json
  {
    "status": "ok",
    "filepath": "fastapi/routing.py",
    "content": "...",
    "start_line": 1,
    "end_line": 150,
    "total_lines": 420,
    "is_truncated": true
  }
  ```

#### 5. `edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str`
* Replaces `old_string` with `new_string` in `/workspace/<filepath>`.
* **Budget Gated:** Yes.
* **3-Tier Resilient Engine (`apply_replacement`):**
  1. `exact`: Character-for-character substring match.
  2. `flexible`: Line-by-line whitespace-stripped match, auto-reindenting `new_string`.
  3. `regex`: Tokenizes code delimiters `( ) : [ ] { } > = <` with flexible `\s*` whitespace matching.
* *Return Schema:*
  ```json
  {
    "status": "ok",
    "filepath": "fastapi/routing.py",
    "occurrences": 1,
    "strategy": "exact",
    "diff": "--- a/fastapi/routing.py\n+++ b/fastapi/routing.py\n...",
    "is_truncated": false
  }
  ```

#### 6. `write_file(filepath: str, content: str) -> str`
* Creates or overwrites a file inside `/workspace`.
* **Budget Gated:** Yes.
* *Return Schema:*
  ```json
  {"status": "ok", "filepath": "reproduce_issue.py", "size": 340}
  ```

---

### 5.3. Graph & Semantic Tools (`swegemma/tools/graph.py`)

#### 7. `search_similar_code(query: str, k: int = 10) -> str`
* Finds top-$k$ symbol nodes with highest cosine similarity to `query` using pre-computed embeddings.
* Pass short or dotted symbol paths (e.g. `"fastapi.routing.serialize_response"` or `"RequestValidationError"`).
* **Budget Gated:** Yes. Total code characters capped at 5,000 chars.
* *Return Schema:*
  ```json
  {
    "status": "ok",
    "query": "serialize_response",
    "results": [
      {"node_name": "fastapi.routing.serialize_response", "code": "def serialize_response...", "similarity": 0.9412}
    ],
    "count": 10,
    "is_truncated": false
  }
  ```

#### 8. `get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str`
* Finds incoming and outgoing AST neighbors of a symbol node in the call graph.
* `edge_type` can be `"calls"`, `"imports"`, or `"references"`.
* **Budget Gated:** Yes. Output capped at 5,000 chars.
* *Return Schema:*
  ```json
  {
    "status": "ok",
    "node": "fastapi.routing.serialize_response",
    "neighbors": ["fastapi.routing._prepare_response_content", "fastapi.utils.create_cloned_field"],
    "count": 2,
    "is_truncated": false
  }
  ```

#### 9. `get_code_subgraph(nodes: list[str]) -> str`
* Extracts the induced subgraph (nodes and interconnecting edges) connecting a list of symbols.
* **Budget Gated:** Yes.

---

## 6. Official Sample Submission Architecture (Multi-LoRA & Prompts)

Inspection of `build/gemma-4-developer-agent/sample_submission` demonstrates Google's official declarative multi-agent and multi-LoRA configuration.

### 6.1. Declarative Routing in `agent.yaml`

```yaml
name: swe_baseline_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora                         # Code editing and diff specialist
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
```

---

### 6.2. Delegated Analysis in `sub_agents/code_analyzer.yaml`

```yaml
name: code_analyzer_agent
description: Analyzes repository source files and symbol graphs to locate root causes.
model: gemma-4-31b-it-qat-w4a16-ct          # Must match base model exactly!
adapter: tool_lora                         # Dedicated LoRA for graph navigation
instruction: !include ../prompts/analyzer.md
tools:
  - read_file
  - search_similar_code
  - get_code_neighbors
  - get_code_subgraph
generate_content_config: !include ../configs/sampling.yaml
```

---

### 6.3. Inference & Thinking Config in `configs/sampling.yaml`

```yaml
temperature: 0.2
top_p: 0.95
max_output_tokens: 16384
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
```

* `thinking_budget: 4096`: Enables Gemma 4 native reasoning tags (`<|channel>thought ... <channel|>`).
* `include_thoughts: true`: Retains thought history across turns for complex reasoning.

---

### 6.4. Official System Prompt Directives (`prompts/system.md`)

The official system prompt reveals critical operational rules:
1. **Speed & Turn Ceiling:** Solve the issue in under **8–10 turns**.
2. **Immediate Localization:** Extract symbol names directly from problem statements and use `search_similar_code` rather than disk sweeps.
3. **Targeted Pytest Only:** Run **only** the single test file or method:  
   `pytest tests/test_target.py -k test_feature`
4. **FastAPI Docs Caveat:** For documentation tasks in FastAPI, edit executable code under `docs_src/`.
5. **Immediate Patch Submission:** Once the targeted test passes, call `submit_patch` immediately and ensure `patch_size > 0`.

---

## 7. The 4 Fatal Anti-Patterns & Instant Disqualification Risks

Extracted directly from `HARNESS_README.md` and `system.md`, these four mistakes cause immediate evaluation failure:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE 4 FATAL SUBMISSION ANTI-PATTERNS                 │
├────────────────────────────────────────────────────────────────────────┤
│ 1. NEVER MODIFY FILES UNDER tests/                                     │
│    Container B runs: git checkout HEAD tests/                          │
│    All test edits are WIPED before grading. Modifying tests causes     │
│    an immediate automatic task failure.                                │
│                                                                        │
│ 2. NEVER RUN BARE pytest OR pytest .                                   │
│    Full repository test suites take 5–10 minutes, cause catastrophic    │
│    command timeouts (300s limit), and deplete turn budgets. Always     │
│    specify the exact target test file.                                 │
│                                                                        │
│ 3. NEVER ATTEMPT TO FIX PRE-EXISTING BROKEN TESTS                      │
│    Many open-source repos have failing tests due to missing external   │
│    fixtures. Ignore pre-existing failures; verify ONLY your bug.       │
│                                                                        │
│ 4. NEVER SEARCH OUTSIDE /workspace                                     │
│    All dependencies are pre-installed in /wheels. Searching            │
│    /usr/local/lib or /opt burns precious tool turns for zero gain.     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 8. The Winning Master Strategy: Scaling LoRA SFT to 0.82+ Pass@1

### 8.1. Why Reinforcement Learning is Sub-Optimal for this Task

* **Interactive RL Bottleneck:** Running online multi-turn RL rollouts requires spinning up Docker containers, checking out commits, installing wheels, and running Pytest on every episode. Each step takes 1–3 minutes. Training 50,000 RL episodes would take months of compute.
* **The SFT Advantage:** In SWE-bench style benchmarks, **Teacher-Forced Supervised Fine-Tuning (SFT)** on expert demonstration trajectories establishes the conditioned policy in hours, delivering predictable, state-of-the-art results without environment instability.

---

### 8.2. Phase 1: Mining the 129 Public Tasks into Gold Trajectories

Because we possess the ground-truth `patch` and `test_patch` for all 129 tasks, we know the exact ground-truth symbol changed (e.g. `fastapi.routing.serialize_response`).

We can programmatically construct the **Optimal Teacher-Forced Trajectory** for each task:
```text
Turn 1: Model calls search_similar_code(keywords_from_issue)
        Output: Top candidate nodes including ground-truth node.
Turn 2: Model calls get_code_neighbors(ground_truth_node)
        Output: Callers and callees from the AST graph.
Turn 3: Model calls read_file(ground_truth_file, start_line, end_line)
        Output: The 50 lines surrounding the target bug.
Turn 4: Model calls edit_file(ground_truth_file, old_string, new_string)
        Output: Success: 1 occurrence replaced (exact patch diff).
Turn 5: Model calls run_command("pytest tests/test_target.py -k test_feature")
        Output: === 1 passed in 0.15s ===
Turn 6: Model calls submit_patch()
        Output: {"status": "ok", "patch_size": X, "files_changed": 1}
```

---

### 8.3. Phase 2: Archetype-Targeted GitHub PR Mining

To scale beyond the 129 public tasks and generate **20,000+ SFT samples**, we run automated commit mining on GitHub repositories matching the competition archetypes:

```text
Target Repositories for Mining:
├── ASGI & Web Frameworks:     FastAPI, Starlette, AIOHTTP, Sanic, Tornado
├── Typing & Schemas:          Pydantic, Pydantic-Core, Attrs, Cattrs, Marshmallow, Msgspec
├── HTTP & Networking:         Requests, HTTPX, HTTPCore, URLLib3, Yarl
├── CLI & Terminal Layout:     Rich, Textual, Click, Typer
└── Developer Tools:           Pytest, Coverage, Pluggy
```

#### Mining Algorithm:
1. Scan git history for commits that touch both `src/*.py` (implementation) and `tests/test_*.py` (test reproducer).
2. Filter for commits where the commit message references a closed GitHub issue (e.g. `Fixes #1234`).
3. Ensure diff lines are small ($\le 50$ lines changed across $\le 2$ files).
4. Run NetworkX AST extraction to generate the call graph for that commit.

---

### 8.4. Phase 3: Teacher-Forced Synthetic Trajectory Synthesis

For every mined commit `(issue, base_commit, ground_truth_patch)`:
* Construct the 6-turn interaction trajectory using the exact tool signatures of `search_similar_code`, `get_code_neighbors`, `read_file`, and `edit_file`.
* Add **self-correction trajectories (10–15% of dataset):**
  * Turn 4 introduces a slight syntax error or failing assertion.
  * Turn 5 runs `pytest` and returns the `AssertionError` traceback.
  * Turn 6 calls `edit_file` with the corrected fix.
  * Turn 7 runs `pytest` and verifies that tests pass.
  * *Why this works:* Teaches Gemma 4 not to panic or terminate on test failures, instilling resilient self-repair behaviors.

---

### 8.5. Multi-LoRA Dual-Specialization Architecture

We will fine-tune **two specialized LoRA adapters** targeting `gemma-4-31b-it-qat-w4a16-ct`:

```
┌────────────────────────────────────────────────────────────────────────┐
│               MULTI-LoRA DUAL SPECIALIZATION ON 4x L4                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Adapter 1: tool_lora (Search & Localization Specialist)              │
│   • Attached to: code_analyzer_agent (sub-agent)                       │
│   • Tools: search_similar_code, get_code_neighbors, read_file         │
│   • Trained on: Converting problem descriptions into precise symbol    │
│     queries, AST neighbor traversal, and pinpointing line ranges.      │
│   • Size: Rank r=32 (~220 MB safetensors)                              │
│                                                                        │
│   Adapter 2: main_lora (Surgical Code Patch Specialist)                │
│   • Attached to: swe_baseline_agent (root agent)                       │
│   • Tools: edit_file, run_command, submit_patch                        │
│   • Trained on: Synthesizing exact old_string/new_string replacements, │
│     interpreting pytest error tracebacks, and self-correcting diffs.   │
│   • Size: Rank r=32 (~220 MB safetensors)                              │
│                                                                        │
│   Total Footprint: ~440 MB (Well below the < 3 GiB limit!)             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Conclusion & Actionable Milestones

1. **Schema Standardization:** Update all pipeline tools in `pipeline/src/agent/tools.py` to match the exact 9 tools (`swegemma.tools`) and signatures.
2. **Trajectory Mining Script:** Implement `pipeline/src/data/mine_trajectories.py` to convert `tasks.jsonl` + `graphs/` into initial Gold SFT trajectories.
3. **LoRA SFT Training Pipeline:** Configure PyTorch + PEFT to fine-tune `tool_lora` and `main_lora` on `gemma-4-31b-it-qat-w4a16-ct`.
4. **Packaging Automation:** Build and validate `submission.zip` matching Google's `sample_submission` blueprint.
