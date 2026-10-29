# Resources & Research Catalogue

This catalogue tracks everything the team reads and references for the **Google Gemma 4 Developer Agent Competition**.
Local PDFs, scraped blog HTML, and external repo clones are stored on disk for reading, but **gitignored** to prevent repository bloat and licensing issues.

---

## 📚 Master Index

| Priority | Source / arXiv | Title | Focus / Read For | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Must** | arXiv:2405.15793 | *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* | ACI design, specialized editor tools, guardrails | Read |
| **Must** | arXiv:2404.04125 | *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* | Benchmark evaluation format, patch synthesis, error modes | Read |
| **Must** | arXiv:2608.23552 | *Prime Agent: A Self-Improving RLM Harness* | Agent-as-a-Tool multi-agent decomposition, execution loops | Read |
| **Must** | Official Docs | *Google Gemma 4 Model Card & Technical Report* | Architecture, QAT w4a16 quantization specs, context limits | Read |
| **Should** | arXiv:2401.03444 | *Executable Code Actions Elicit Better LLM Agents* | Action spaces in coding agents vs plain text generation | Planned |
| **Should** | Blog / Guide | *vLLM PagedAttention & LoRA Multi-Adapter Serving* | Offline serving throughput, KV cache limits, tensor parallelism | Planned |
| **Should** | arXiv:2305.18290 | *Direct Preference Optimization (DPO)* | Preference learning on paired resolved vs failed patches | Planned |

---

## 📁 Local Directory Guide

* `resources/official/` - Official competition specs, rules snapshots, and scoring criteria. (Tracked in Git)
* `resources/papers/` - Downloaded PDFs & extracted texts. (Local only, gitignored)
* `resources/web/` - Scraped web guides and engineering blogs as Markdown. (Local only, gitignored)
* `resources/repos/` - Third-party code clones for inspecting reference implementations. (Local only, gitignored)

