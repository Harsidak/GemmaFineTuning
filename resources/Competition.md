# Competition Details

## Overview:
Today’s best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware?

Post-train an open model into a reliable agent that navigates complex codebases and drafts fixes for real software issues, accelerating developer workflows on everyday hardware.

## Evaluation:

Based on the submission zip file
```
submission.zip
├── agent.yaml                  # REQUIRED: Root agent config
├── configs/
│   └── sampling.yaml           # Optional: Generation parameters loaded via !include
├── prompts/
│   ├── system.md               # Optional: System instructions loaded via !include
│   └── analyzer.md
├── sub_agents/
│   └── code_analyzer.yaml      # Optional: Sub-agent or AgentTool YAML configurations
├── adapters/                   # Optional: Fine-tuned PEFT LoRA adapters or model weights
│   ├── main_lora/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── tool_lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── skills/                     # Optional: ADK Skill directories
    └── repo_navigation/
        ├── SKILL.md
        ├── scripts/            # Python or bash scripts executed in a sandbox
        └── resources/          # Domain knowledge markdown files
```

## Scoring:
Issue Scoring
Similar to SWE-Bench, your agent's submitted patches are evaluated by a PASS/FAIL metric. For each issue, the submitted patch is applied to that issue's code repository and that issue's validation tests are run. Your submission's overall score is the percentage of patched repositories that pass the validation tests.

Your agent has a limit of 12 hours to submit patches for all tasks, inclusive of sandbox setup time, but excluding time for patch validation. You may, but are not required, to set per-task time limits in your submission's eval_config.yaml.

## Model Selection, Budget and Harness

### Model selection and LoRa 
At present, we only support the following Gemma 4 model variants:

- gemma-4-31b-it-qat-w4a16-ct
- gemma-4-12b-it-qat-w4a16-ct
- gemma-4-e4b-it-qat-w4a16-ct
- gemma-4-e2b-it-qat-w4a16-ct

Every agent and subagent in your submission must use the same model. It is not possible to mix models within the same submission.

You may, but are not required, to include one or more .safetensors format LoRA adapters with your submission. While you are restricted to a single base model, you may use different adapters for each agent in your submission:

Place PEFT LoRA directories (containing adapter_config.json and adapter_model.safetensors) inside adapters/<adapter_name>/.
Reference adapter: <adapter_name> on any LlmAgent in agent.yaml or sub_agents/*.yaml (for example, adapter: main_lora on the root coder agent and adapter: tool_lora on a read-only analyzer AgentTool).
### Tool, Prompt, and Skill Rules
- !include Directives: Resolves file paths relative to the directory of the file containing the tag. For instance, in agent.yaml, !include prompts/system.md loads prompts/system.md.
- Sandboxing: Path traversal outside the submission root (e.g., via ../ or symlinks) is not allowed.
- Allowed Tools: Your agent can only request tools provided by the competition harness or custom subagents defined via agent_tool.
- Skill Structure & Sandboxing: Each skill must be a directory containing a SKILL.md manifest with YAML frontmatter (name: <skill-name>). Scripts executed via run_skill_script run securely inside the competition's persistent Docker container, sharing filesystem access with run_command and debiting execution time against your central budget. Agents can inspect domain knowledge files using load_skill_resource.

### The predefined tools available to your agent are:

- run_command(command: str) -> str - Executes a shell command in /bin/bash -c inside /workspace.
- submit_patch() -> str - Stages untracked file intents (git add -N .) and captures git diff HEAD from /workspace.
- get_status() -> str - Returns live budget consumption and patch status.
- read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str - Reads a file from /workspace with 1-indexed inclusive line slicing.
- edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str - Replaces old_string with new_string in an existing non-empty file inside /workspace.
- write_file(filepath: str, content: str) -> str - Creates or overwrites a file at /workspace/<filepath>, automatically creating parent directories (mkdir -p).
- get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str - Finds incoming and outgoing neighbors of a symbol (node) in the repository call/dependency graph.
- search_similar_code(query: str, k: int = 10) -> str - Finds top-k graph nodes with highest cosine similarity to query in the pre-computed embeddings.
- get_code_subgraph(nodes: list[str]) -> str - Extracts the induced subgraph (all nodes and interconnecting edges) for a list of symbols.
- See the HARNESS_README.md file in the dataset for detailed information about the submission format and execution environment.
## Prices:
The Gemma 4 Developer Agent Competition: $65,000

1st Place: $37,000
2nd Place: $18,000
3rd Place: $10,000

## Bonus Prize: Small Model Development: $10,000

We are offering an additional $10,000 bonus prize dedicated to small-footprint developer agents to the highest-ranking submission utilizing Gemma E2B or Gemma E4B. Same submission rules and tie-breaking apply.

Eligibility: Submissions must explicitly use either Gemma E2B or Gemma E4B as the base model. For a submission to be eligible it must be one of the team’s two selected final submissions. LoRA fine-tuning is allowed just as done for the main track.

Daily Leaderboard Tracking: A separate Kaggle notebook will update daily to display filtered standings for eligible small model entries.

Cross-Track Eligibility: Submissions eligible for the Small Model Prize remain fully eligible for general leaderboard awards and the Paper Track.

Deadline: Same submission deadlines apply as the main track prizes.

Top submissions will be highlighted at a Google hosted event (such as a NeurIPS expo workshop) and awarded prizes. Submissions leveraging the accompanying released graph and embedding dataset are highly encouraged, but not a requirement for submission. Submissions must be made by November 12, 2026 at 11:59PM UTC.

## Dataset Description(which is inside the Build directory of this repo ~ build folder is gitignored)
This competition evaluates AI software engineering agents on their ability to solve real-world Python bug fixes and feature requests.

To assist agents in large codebase navigation, dependency understanding, and semantic retrieval, the competition provides code graphs, graph/node embeddings, and specialized graph tools alongside traditional repository exploration mechanisms.

Participants are provided with a training set (including tasks, code graphs, and embeddings) for local agent development, prompt engineering, and offline validation. Submissions are evaluated against a test set curated under a similar pipeline, with identical graph generation methods, and verification standards.

In this public-facing version of the dataset, you will only see the training set. When your submission is scored, this training set will be replaced with the hidden test set.

Files
tasks.jsonl

Line-delimited JSON file containing the 129 public development tasks. Each JSON object represents one bug-fixing benchmark instance and includes both the reference fix (patch) and verification unit tests (test_patch) for local eval\ uation.

instance_id - unique identifier for the benchmark task combining the repository short name and issue or pull request number (for example, fastapi_11194 or rich_3454).
repo - source GitHub repository in owner/repo format (fastapi/fastapi, Textualize/rich, psf/requests, or encode/httpx).
base_commit - 40-character hexadecimal Git commit SHA representing the repository snapshot immediately prior to the bug fix.
problem_statement - natural-language bug report or feature issue description provided to the agent as the task prompt.
hints_text - optional issue comments or maintainer hints providing additional context (may be empty).
patch - reference unified git diff solution patch that resolves the issue on base_commit (provided in published/tasks.jsonl for local training and analysis; excluded from the hidden rerun/tasks.jsonl test set).
test_patch - unified git diff patch containing the verification unit tests executed by pytest during Phase 2 grading (provided in published/tasks.jsonl for local verification; kept in the private secret/solution.parquet archive during competition scoring).
created_at - ISO-8601 timestamp indicating when the issue or pull request was created.
snapshots/ Directory containing 129 compressed Git working tree archives (snapshots/<instance_id>.tgz, such as fastapi_11194.tgz, requests_7205.tgz, rich_2725.tgz, and httpx_3672.tgz). Each archive contains a full Git repository frozen at base_commit with forward commit history removed so agents can safely run git status, git log, and git diff inside /workspace without leaking future fixes.

graphs/ Directory containing serialized NetworkX Abstract Syntax Tree (AST) call and dependency graphs in JSON format (256 files total: 129 task-named files <instance_id>.json hard-linked to 127 commit-named files <repo_short>_<base_commit>.json). These graphs back the get_code_neighbors and get_code_subgraph agent tools and use the following schema:

directed - boolean flag indicating whether graph edges are directed (true).
multigraph - boolean flag indicating whether multiple edges can connect the same pair of symbols (true).
graph - dictionary of global graph metadata attributes.
nodes - list of code entity node objects in the repository snapshot.
nodes[*].id - unique fully qualified Python symbol path for the module, class, or function (for example, fastapi.routing._prepare_response_content).
nodes[*].name - human-readable qualified symbol name matching id.
nodes[*].text - full Python source code definition of the function, method, or class at base_commit.
edges - list of directed structural relationship objects between code nodes.
edges[*].source - origin symbol id where the call or reference originates.
edges[*].target - destination symbol id being called, imported, or referenced.
edges[*].type - structural relationship category linking source to target (such as calls).
edges[*].key - integer index ($0, 1, \dots$) distinguishing parallel edges in the multigraph.
embeddings/ Directory containing compressed NumPy archives (256 files total: 129 task-named files <instance_id>.npz hard-linked to 127 commit-named files <repo_short>_<base_commit>.npz). These archives power offline semantic code retrieval via the search_similar_code tool.

<node_id>.npy - float32 NumPy array of length 256 containing dense semantic vector features embedding_[0-255] for the corresponding AST graph symbol node_id.
wheels/ Directory of 124 pre-compiled offline Python binary wheels (*.whl) covering historical dependency versions for fastapi, starlette, pydantic, requests, urllib3, rich, httpx, httpcore, pytest, and transitive packages. Mounted read-only at /wheels/ inside the sandbox container so pip can install repository dependencies without internet access.

docker/ Container build specifications and Python standard library compatibility shims used to build the local swebench-sandbox:latest image:

Dockerfile.sandbox - base sandbox container specification providing Python 3.13, git, pytest, and build backends (setuptools, hatchling, flit-core, poetry-core, pdm-backend).
Dockerfile.public - public development container variant pre-configured for the 129 open-source tasks.
imp.py - compatibility shim restoring the legacy imp module removed in Python 3.13 for older test suites.
telnetlib.py - compatibility shim restoring the legacy telnetlib module removed in Python 3.13.
sandbox/setup.py Container initialization script executed inside /workspace when preparing a task sandbox. Inspects pyproject.toml, setup.cfg, or requirements.txt, resolves compatible offline wheels from /wheels/, performs an editable install (pip install --no-index --find-links=/wheels -e .), and creates a clean baseline Git commit so git diff HEAD captures only your agent's edits.

sample_submission/ Ready-to-run baseline ADK submission directory illustrating the declarative YAML schema, multi-agent delegation, and multi-LoRA routing.

HARNESS_README.md In-depth technical guide covering the swegemma, adk-submission, and adk-eval-core libraries, the two-container sandbox lifecycle, tool call signatures, context compaction, and local CLI evaluation commands.

submission.parquet Output file automatically produced at /kaggle/working/submission.parquet during evaluation when the inference script runs your uploaded submission.zip archive against the benchmark tasks:

id - unique task identifier matching instance_id in tasks.jsonl.
prediction - unified git diff patch string generated by your agent relative to base_commit (or NO_PATCH if no changes were produced).
Dataset Generation & Verification Pipeline
Both the training set and test set were produced using an automated curation and two-phase verification harness:

Commit & PR Mining: Filtered repository histories for commits that jointly modify core functional logic (.py) and corresponding unit tests (test_*.py / *_test.py).
Issue Linking & Prompt Extraction: Matched each code change with its original problem description, issue tracker thread, or bug report.
Filtering & De-noising: Excluded mechanical refactors, documentation-only changes, large-scale automated changes (LSCs), and ambiguous issues lacking reproducible acceptance criteria.
Execution Verification Harness:
Fail-to-Pass (Baseline Verification): The test reproducer (test_patch) is applied to base_commit without the fix; tests must fail (non-zero exit code).
Pass-to-Pass (Solution Verification): The reference fix (patch) is applied with test_patch; the entire test suite must pass cleanly (exit code 0).
Near Completion with Frontier Models: For the test set, we check that a larger frontier model can pass the case or get within a single test case of passing.
There are about 120 tasks in the test set, evenly divided between the public and private splits. The test set was curated from a set of private repositories.


## kaggle Constraint:
1) 12 hour window
2) 96 gb vram provided(total of all 4 l4s): This competition has access to Kaggle's pool of powerful new L4x4 machines!
3) No internet allowed