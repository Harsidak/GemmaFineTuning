# Resources & Research Catalogue

This catalogue tracks the core research literature, open-source pipelines, trajectory datasets, and technical specifications powering our **Google Gemma 4 Developer Agent Competition** system.
Local PDFs, detailed technical notes, and external repo clones under `resources/{papers,web,repos}/` are gitignored to keep the repository lightweight, while each local artifact has a `<filename>.meta.yaml` sidecar following `resources/_TEMPLATE.meta.yaml`.

---

## 📚 1. Master Paper & Specification Index

### A. Gemma 4 Architecture, Quantization & Serving
| Priority | Source / arXiv | Title | Focus / Key Takeaway for Competition | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Must** | Official Docs | *Google Gemma 4 Model Card & Technical Report* | Multimodal `Gemma4ForConditionalGeneration` (`language_model` prefix), `hidden_size=5376` (31B), QAT `w4a16-ct` specs | Read |
| **Must** | vLLM / Unsloth | *Gemma 4 Tool-Calling & Reasoning Wire Grammar (`tool_chat_template_gemma4.jinja`)* | Exact `<\|turn>`, `<\|channel>thought...<channel\|>`, `<\|tool_call>call:fn{k:<\|"\|>v<\|"\|>}<tool_call\|>`, and `response:fn{...}` DSL | Read |
| **Must** | `build/` Docs | *`swegemma` + `adk-submission` + `adk-eval-core` (`HARNESS_README.md`)* | Two-container hermetic grading, 4x L4 vLLM flags, offline `.npz` cosine lookup mechanics, ADK `skills/` execution | Read |
| **Should** | arXiv:2312.03732 | *rsLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* | Scaling behavior across ranks $r \in \{16, 32, 64\}$ (verify vLLM `-ct` compatibility before toggling `use_rslora`) | Read |

### B. Multi-Agent Specialization, Agentic SFT & Preference Post-Training (Pillar 1)
| Priority | Source / arXiv | Title | Focus / Key Takeaway for Competition | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Must** | arXiv:2403.12881 | *Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for LLMs* | Strict assistant-only loss masking (never compute loss on tool stdout), format/reasoning decomposition, negative turn recovery | Read |
| **Must** | arXiv:2310.05915 | *FireAct: Toward Language Agent Fine-tuning* | Multi-task trajectory mixing and why specialist sub-agent policies outperform monolithic single-prompt loops | Read |
| **Must** | arXiv:2403.02502 | *ETO: Trial and Error — Exploration-based Trajectory Optimization for LLM Agents* | Iterative SFT $\rightarrow$ on-policy failure collection $\rightarrow$ Trajectory DPO (`chosen` resolved vs `rejected` anti-pattern rollouts) | Read |
| **Must** | arXiv:2502.18449 | *SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution* | Dense similarity rewards on unified diffs + disciplined compact `<\|channel>thought` reasoning before tool invocation | Read |
| **Should** | arXiv:2608.23552 | *Prime Agent: A Self-Improving RLM Harness* | Agent-as-a-Tool (`agent_tool`) context isolation (`skip_summarization: true`) to prevent 32k context overflow | Read |
| **Should** | arXiv:2305.18290 | *Direct Preference Optimization (DPO)* | Offline preference alignment without online Docker sandbox rollouts | Read |

### C. Code Graph Localization, Evaluation & Hybrid Verifiers (Pillar 2)
| Priority | Source / arXiv | Title | Focus / Key Takeaway for Competition | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Must** | arXiv:2503.09089 | *LocAgent: Graph-Guided LLM Agents for Code Localization* | Multi-hop AST/dependency traversal (`search_similar_code` $\rightarrow$ `get_code_neighbors`) achieving >92% file recall in 2–3 hops | Read |
| **Must** | arXiv:2410.14684 | *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph* | $k$-hop ego-graph retrieval (`get_code_subgraph`) around candidate definitions & test callers | Read |
| **Must** | arXiv:2407.01489 | *Agentless: Demystifying LLM-based Software Engineering Agents* | Hierarchical localization $\rightarrow$ surgical `edit_file` replacement $\rightarrow$ regression-aware test filtering | Read |
| **Should** | arXiv:2408.03910 | *CodexGraph: Bridging LLMs and Code Repositories via Code Graph Databases* | Translating conceptual bug reports into symbol-level graph queries rather than raw prose queries | Read |
| **Should** | arXiv:2405.15793 | *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* | Windowed file viewing (150-line cap), linter-gated edits, error-message ergonomics | Read |
| **Should** | arXiv:2404.04125 | *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* | Fail-to-Pass & Pass-to-Pass two-phase container evaluation foundations | Read |

### D. High-Volume Verified SWE Data Generation & Decontamination (Pillar 3)
| Priority | Source / arXiv | Title | Focus / Key Takeaway for Competition | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Must** | arXiv:2412.21139 | *SWE-Gym: Training Software Engineering Agents and Verifiers* | 2,438 execution-verified Python tasks + training SFT trajectories and execution-free outcome verifiers | Read |
| **Must** | arXiv:2504.07164 | *R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents* | **SWE-GEN** commit back-translation recipe (8.7k+ executable Python tasks without requiring clean GitHub issues) | Read |
| **Must** | arXiv:2504.21798 | *SWE-smith: Scaling Data for Software Engineering Agents* | Synthesizing 50,000+ execution-verified Fail-to-Pass bugs inside a single pre-built repo environment via AST/LLM bug injection | Read |
| **Must** | arXiv:2505.20411 | *SWE-rebench: Automated Pipeline for Task Collection & Decontaminated Evaluation* | Continuous mining of fresh 2025–2026 Python PRs + LLM quality scoring for self-contained problem statements | Read |
| **Should** | arXiv:2501.05040 | *SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution* | Coarse-to-fine retrieval + surgical editing decomposition for high-efficiency open models | Read |

---

## 🛠️ 2. Open-Source Pipelines, Repos & HuggingFace Datasets

| Resource / Repo | URL / Identifier | Use in Our Factory |
| :--- | :--- | :--- |
| **SWE-smith** | `https://github.com/SWE-bench/SWE-smith` | Unlimited Fail-to-Pass bug synthesis (AST mutations, PR mirroring, LLM bug injection) across 30+ Python repos |
| **R2E-Gym** | `https://github.com/R2E-Gym/R2E-Gym` | 8.7k procedurally verified Python commit environments + SWE-GEN back-translated problem statements |
| **SWE-Gym** | `https://github.com/SWE-Gym/SWE-Gym` | Verified Python task instances + trajectory filtering & verifier training scripts |
| **LocAgent** | `https://github.com/gersteinlab/LocAgent` | Reference patterns for training `tool_lora` on multi-hop AST/call-graph localization |
| **RepoGraph** | `https://github.com/ozyyshr/RepoGraph` | AST definition-reference & ego-subgraph construction matching `swegemma` `graphs/*.json` |
| **Nebius Trajectories** | `hf://nebius/SWE-rebench-openhands-trajectories` | Reference multi-turn error-recovery traces on fresh decontaminated Python PRs |
| **Unsloth (Gemma 4)** | `https://github.com/unslothai/unsloth` | 2x faster, 60% lower VRAM LoRA/QLoRA training for Gemma 4 12B/31B/E4B on Kaggle L4/T4 & Colab A100 |
| **TRL (`SFTTrainer` / `DPOTrainer`)** | `https://github.com/huggingface/trl` | Assistant-only completion loss masking (`DataCollatorForCompletionOnlyLM` / token-span labels) & Trajectory DPO |

---

## 📁 3. Tracked Research Reports & Local Directory Guide

* [`resources/initial_data_inspection.md`](file:///d:/FORGE/Research/GemmaFineTuning/resources/initial_data_inspection.md) — Forensic EDA of `tasks.jsonl`, patch geometry, and harness overview. (Tracked in Git)
* [`resources/research_training_eval_data.md`](file:///d:/FORGE/Research/GemmaFineTuning/resources/research_training_eval_data.md) — **Deep 3-Pillar Research Blueprint**: Multi-Agent LoRA Training on Kaggle, 4-Tier Diagnostic Evaluation System, and 5-Engine High-Signal Data Factory. (Tracked in Git)
* [`resources/lora_sft_playbook.md`](lora_sft_playbook.md) — **LoRA SFT Playbook for Gemma 4 on Kaggle** (2026-10-29): L4×4 vs T4 hardware, the official scoring stack (vLLM 0.19.1, transformers 5.13.1), memory math for the 262k vocab, train/serve parity (template, weights, vLLM LoRA on int4), LoRA hyperparameters (*LoRA Without Regret*), time-budgeted 12 h runs, SGLang vs vLLM, Unsloth vs TRL. (Tracked in Git)
* `resources/papers/` — Synthesized paper notes & `.meta.yaml` sidecars for `SWE-Gym`, `R2E-Gym`, `SWE-smith`, `LocAgent`/`RepoGraph`, and `Agent-FLAN`/`ETO`. (Local only, gitignored)
* `resources/web/` — Gemma 4 vLLM QAT W4A16 LoRA & `tool_chat_template_gemma4.jinja` wire grammar notes + `.meta.yaml`. (Local only, gitignored)
* `resources/repos/` — Open-source SWE pipeline architecture notes + `.meta.yaml`. (Local only, gitignored)
