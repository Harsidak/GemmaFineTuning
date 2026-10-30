# LoRA SFT Playbook for Gemma 4 on Kaggle

Research notes for building one robust, re-runnable LoRA SFT pipeline for the Gemma 4 Developer Agent competition.
Written 2026-10-29. Every claim is tagged:

- **[verified]**: checked against a primary source or our own files.
- **[source]**: from a published source we have not reproduced.
- **[estimate]**: back-of-envelope math, to be replaced by a measurement.
- **[unverified]**: plausible but unchecked, so test it before relying on it.

---

## 0. TL;DR

1. **Train on the 4×L4 machine, not T4.** Notebooks attached to this competition can request `"machine_shape": "NvidiaL4"`, which gives 4×L4 = 96 GB, bf16, FlashAttention-capable GPUs. That is the same hardware the scorer uses. **[verified]** from the official starter's `kernel-metadata.json`.
   - Those sessions have **no internet**.
   - They burn GPU quota at **2×** the T4 rate. **[source]** competition overview.
   - Budget for about **one full 12-hour L4 run per week**.
2. **Only vLLM matters for inference here, and SGLang does not help SFT.** SFT training does not use an inference server at all. The scorer runs **vLLM 0.19.1** with the `gemma4` tool-call and reasoning parsers. Our in-notebook eval must use the same stack, or its score cannot be trusted.
3. **The biggest risk is train/serve mismatch, not hyperparameters.** There are three places it can happen:
   - **Chat template.** The Kaggle `q4_0-unquantized/2` template differs from the served `w4a16-ct` template. **[verified]** `DataMining/template_check/report.json`.
   - **Weights.** We train on bf16 or NF4, but serving uses int4 compressed-tensors.
   - **LoRA application.** vLLM applies LoRA on top of an int4 compressed-tensors base, and an open vLLM issue reports rank-32 all-layer adapters misbehaving there. **[source]** vllm#50059.
   - The pipeline must *measure* each of these, not assume them away.
4. **Gemma 4's 262,144-token vocabulary is the memory killer.** Full logits for one 16k-token sequence take 8.6 GB in bf16, and 17 GB if upcast to fp32. Compute logits only where the loss is taken, or use a fused or chunked cross-entropy. **[estimate]** arithmetic in §3.
5. **Our LoRA learning rate is probably 5–10× too low.** `train_lora_sft.yaml` uses `2e-5`. LoRA's optimal LR is about 10× full fine-tuning's, typically `1e-4`–`2e-4`. **[source]** Thinking Machines, *LoRA Without Regret*.
6. **Apply LoRA to every linear layer of the *language model only*.** MLP layers matter most. Plain `q_proj`…`down_proj` names also match the vision tower's layers, so use a regex that pins the `language_model` prefix.
7. **Make the run time-aware, not epoch-aware.** Use a schedule that decays the LR when the *clock* says so (warmup → stable → decay). Checkpoint by wall-clock time. Always leave the last hour for export, eval and packaging.
8. **A 3-task eval is a smoke test, not a score.** A model with a true 30% pass rate shows 0/3 a third of the time. Use it to prove the adapter loads in vLLM and the agent produces patches. Use val loss and the Evaluation track's full runs to compare models.
9. **Iterate on E4B, then spend L4 runs on 31B.**
   - E4B fits on one L4, so one L4×4 session can train **four configs in parallel**, one per GPU.
   - 31B needs all four GPUs for one run. **[estimate]** It only gets through roughly 500–700 trajectory passes in 10 hours with a simple model split.
10. **Use TRL + PEFT + Accelerate as the backbone** (official, stable, multi-GPU), with Liger or a custom loss for the vocabulary problem. Unsloth is faster on one GPU but monkey-patches heavily, has weak multi-GPU support, and must be wheel-pinned offline. Keep it as an optional single-GPU backend at most.

---

## 1. What we actually run on

### 1.1 Kaggle accelerators

| Option (`machine_shape`) | GPUs | VRAM | bf16 | FlashAttention 2 | Internet | Quota cost |
|---|---|---|---|---|---|---|
| `NvidiaTeslaT4` (UI: "GPU T4 ×2") | 2× T4 (Turing, SM75) | 2×15 GB | no (fp16 only) | no (needs SM80+) | allowed | 1× |
| `NvidiaTeslaP100` | 1× P100 | 16 GB | no | no | allowed | 1× |
| `NvidiaL4` (competition L4×4) | 4× L4 (Ada, SM89) | 4×24 GB = 96 GB | yes | yes | **disabled** | **2×** (may rise) |

- **Session and quota limits** **[source]**:
  - GPU sessions are capped at 12 hours.
  - Weekly GPU quota is "up to" about 30 hours.
  - At 2×, one 12-hour L4 run uses about 24 of those 30 hours.
- **The machine shape string** **[verified]**: `"NvidiaL4"` is what the official starter notebook (`ryanholbrook/getting-started-gemma-4-developer-agent`) uses. That notebook also has `enable_internet: false` and pins a Kaggle `docker_image` digest.
- **Our settings today**: `scripts/kaggle_settings.json` still uses `NvidiaTeslaT4` with internet on.

### 1.2 The scoring software stack (the official wheelhouse, `metric/gemma-4-developer-agent-wheelhouse`) **[verified]**

| Package | Version | Why we care |
|---|---|---|
| `vllm` | 0.19.1 | Serves the model and our LoRAs at scoring time. |
| `transformers` | 5.13.1 | Our training libraries must be compatible with this version. Our local `.venv` has 5.21.0. |
| `compressed_tensors` | 0.15.0.1 | Format of the served int4 weights. |
| `bitsandbytes` | 0.46.1 | For QLoRA. Local `.venv` has 0.50.2. |
| `swegemma` / `adk_submission` / `adk_eval_core` | 0.2.11 / 0.2.13 / 0.1.0 | Harness, server manager, sandbox. |
| `google_adk` | 1.36.1 | Agent runtime. |

- **Not in the wheelhouse**: `torch` (it comes from the pinned Kaggle docker image), `peft`, `trl`, `accelerate`, `datasets`, `liger-kernel`.
- On an L4 session (no internet), any of those missing from the image must come from **our own private wheelhouse dataset**, pinned to match `transformers==5.13.1` and the image's torch. **[unverified]** We have not yet checked which of them the image already ships.

### 1.3 How the official starter runs an eval inside a notebook **[verified]**

The starter notebook already does what we want for the last hour of a run:

1. **Install the wheels.** `pip install --no-deps --force-reinstall` every wheel from the wheelhouse. It first restores the `+cu128` suffix that Kaggle strips from wheel filenames.
2. **Start the server.** `adk_submission.VllmServer(VllmConfig(...))` with these settings:
   - `tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`, `max_model_len=32768`
   - `gpu_memory_utilization=0.90`, `tensor_parallel_size=4`
   - `enable_lora`, `max_loras = number of adapters`, `max_lora_rank = max(detected ranks)`
   - The 12B model needs `--hf-overrides {"architectures": ["Gemma4ForCausalLM"], ...}`.
3. **Run the tasks.** `swegemma.evaluate.Evaluator(EvalConfig(..., sandbox='subprocess', ...))`, which needs no Docker. The `EventsCompactionConfig(token_threshold=14336, ...)` setting is the context compaction the agent will see at scoring time.
4. **Package and validate.** `adk_submission.validate_directory` plus checks on zip size and file extensions.

**Implication:** the eval and packaging stages should *reuse these exact calls*, not reimplement them.

---

## 2. The two models **[verified]** (from `config.json` of the `-qat-w4a16-ct` repos)

| | E4B | 31B |
|---|---|---|
| Architecture | `Gemma4ForConditionalGeneration` (text + vision + audio) | `Gemma4ForConditionalGeneration` (text + vision; audio null) |
| Text layers / hidden size | 42 / 2560 | 60 / 5376 |
| Attention heads / KV heads | 8 / 2 | 32 / 16 |
| Head dim, sliding vs global layers | 256 / **512** | 256 / **512** |
| Sliding window | 512 | 1024 |
| Vocabulary | **262,144** | **262,144** |
| `final_logit_softcapping` | 30.0 | 30.0 |
| Special features | Per-layer embeddings (`hidden_size_per_layer_input=256`), **18 KV-shared layers** | — |
| Quantization | int4, group size 32, `pack-quantized`; `lm_head` and vision/audio parts unquantized | same |

What these features mean for training:

- **Head dim 512 on the global layers.**
  - FlashAttention-2 supports head dims up to 256, so FA2 / "padding-free" training probably cannot cover these layers. **[unverified]** Check what `attn_implementation` transformers picks.
  - On T4, vLLM's Triton attention kernel exceeds Turing's shared memory at head dim 512 and needs a patch. **[source]** dev.to "Gemma 4 on a Tesla T4". This is one more reason to evaluate on L4.
- **Softcapping.** Any custom loss code must apply `30·tanh(logits/30)` before the cross-entropy, or it trains a different function than the model serves.
- **KV-shared layers (E4B).** The last 18 layers reuse K/V from earlier layers, so LoRA on their `k_proj`/`v_proj` may receive no gradient. Check for dead adapters after the first step (§8).

---

## 3. Memory math (why naive SFT OOMs)

**Logits.** For a sequence of length `T`, the full logits take `T × 262,144 × bytes`. **[estimate]**

| T | bf16 logits | fp32 logits | + gradient of logits |
|---|---|---|---|
| 4,096 | 2.1 GB | 4.3 GB | ×2 |
| 16,384 | 8.6 GB | 17.2 GB | ×2 |
| 32,768 | 17.2 GB | 34.4 GB | ×2 |

In agent trajectories, most tokens are tool outputs (file contents, pytest logs), which carry **no** loss. Typically only 10–25% of tokens are assistant tokens. **[estimate]** Measure it in the data preflight. Three fixes, best first:

1. **Logits only at supervised positions.** Run the decoder, gather hidden states where `labels != -100`, then apply `lm_head`, the softcap and the CE on just those rows.
   - It is simple and exact, and cuts logits memory 4–10×.
   - It needs a small custom `compute_loss`.
2. **A fused linear cross-entropy kernel** (Liger `FusedLinearCrossEntropy`, Apple's Cut Cross-Entropy). It never materializes the full logits.
   - Gemma 4 support (including the softcap) is **[unverified]**. Liger lists Gemma 1–3 patches.
3. **Chunked CE.** Compute the loss in slices of the sequence. Unsloth does this internally for Gemma.

**Weights.** **[estimate]**

| Model | bf16 | NF4 (QLoRA) |
|---|---|---|
| E4B (about 8B parameters including per-layer embeddings) | ~16 GB | ~6–8 GB |
| 31B | ~62 GB | ~17–18 GB |

- **Unsloth's published figures** **[source]**: E4B LoRA 17 GB; E4B QLoRA ~10 GB; 31B QLoRA 22 GB. These are at short context.
- **Long context** costs extra activation memory, even with gradient checkpointing.

**What fits where.** **[estimate]** Confirm with a calibration run.

| Model | T4 (15 GB) | One L4 (24 GB) | 4× L4 (96 GB) |
|---|---|---|---|
| E4B QLoRA, 16k context | yes, tight; fp16 only | yes, comfortable | 4 independent runs |
| E4B bf16 LoRA, 16k context | no | tight | yes |
| 31B QLoRA, 16k context | no | no | yes, model split across 4 GPUs |
| 31B bf16 LoRA, 16k context | no | no | maybe (62 GB weights + activations); risky |

---

## 4. Train/serve parity (where agent fine-tunes actually fail)

A LoRA that looks great on loss curves can still score 0. That happens when the text it learned to emit is not exactly what the serving stack parses and feeds back.

### 4.1 Chat template **[verified]**

- **Served E4B template:** `chat_template.jinja` of `google/gemma-4-E4B-it-qat-w4a16-ct`, sha `0a2c8073c878`. It equals TRL's `gemma4_v5.jinja`.
- **Training checkpoint's template:** the Kaggle `gemma-4-e4b-it-qat-q4_0-unquantized/2` ships a different one, sha `33204f1acb5b`, which is TRL's older `gemma4.jinja`.
- **31B and 12B:** the Kaggle parents differ from the served template there too.
- **Rule:** always render training data with the **served** template (`dataset.chat_template_path`). Also record its sha256 in the run manifest.
- **Unsloth's guidance** **[source]**: a template mismatch is "the most common cause of worse exported-model behavior". In multi-turn data, keep only the final visible answer in history and do not feed earlier thought blocks back in. **[unverified]** Whether the ADK/LiteLLM path re-sends previous `reasoning` must be checked against a real `traces/trace_<id>.json`. Then train exactly the way the harness renders.

### 4.2 Which base weights to train against

| Option | What it is | Train/serve gap | Memory |
|---|---|---|---|
| A. QAT bf16 parent (`*-qat-q4_0-unquantized`) + bf16 LoRA | The weights the int4 model was quantized from | Small: int4 rounding only. QAT was trained to make this small. | bf16 |
| B. QAT bf16 parent loaded as NF4 (QLoRA) | Re-quantizes to a *different* 4-bit grid than serving uses | The adapter partly learns to fix NF4 error that does not exist at serving | 4-bit |
| C. The served `w4a16-ct` itself, loaded in transformers (decompressed or `run_compressed`) | Exactly the int4 values vLLM multiplies with | None in the weights | bf16 if decompressed; 4-bit if `run_compressed` works with PEFT **[unverified]** |

- **Recommendation:** start with **A**, or **B** when memory forces it. Add a cheap **parity metric**: compute val loss of the trained adapter on base A *and* on base C (decompressed served weights). A large gap means the adapter is fitting quantization artifacts.
- **Worth a 30-minute spike:** can PEFT attach LoRA to transformers' compressed-tensors `run_compressed=True` linears, and does backward work? If yes, C is the best of both: exact parity at 4-bit memory.

### 4.3 Adapter format and vLLM

- **Target modules.** Use a regex so that only text-model linears get LoRA, for example:
  ```
  r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
  ```
  - Plain names also match `vision_tower.*.q_proj` and its siblings.
  - **[verified]** The sample adapters use keys `base_model.model.model.language_model.layers.{i}.…`.
- **Saving.** Save the adapter in **bf16** (fp16 if trained on T4, then cast) as `adapter_model.safetensors` plus `adapter_config.json`.
- **Rank.** Rank must be ≤ 128. The harness sets `max_lora_rank = max(adapter ranks)`.
- **Size.** Total unpacked submission must be under 3 GiB.
- **Open risk:** vllm#50059 (open, no maintainer response). On a compressed-tensors int4 base (vLLM 0.17.1 and 0.24.0), **rank-32 all-layer** adapters gave weaker and non-deterministic outputs. Rank-8 partial adapters were fine, and the same adapter was fine in llama.cpp. Our scorer is vLLM 0.19.1, inside that range. **[source]**
  - The pipeline needs an **adapter parity check**: run the same few prompts through HF (base + adapter) and vLLM (w4a16-ct + LoRA), then compare next-token log-probs and greedy outputs.
  - If they diverge, drop to r=16 or narrower targets, or merge-then-quantize.

### 4.4 Tool-call wire format

The served model emits `<|tool_call>call:fn{k:<|"|>v<|"|>}<tool_call|>`, and vLLM's `gemma4` tool parser turns that into structured calls. A cheap CPU check catches broken training data before any GPU time:

1. Render each assistant target span with the served template.
2. Run it through the same `gemma4` tool parser vLLM uses.
3. Confirm it yields exactly the trajectory's tool calls.

---

## 5. LoRA hyperparameters that matter

Main source: Thinking Machines, *LoRA Without Regret* (2025), plus the TRL guide that reproduces it.

- **Which layers.** Apply LoRA to **all** linear layers, especially the MLP (`gate/up/down`). Attention-only LoRA underperforms even at matched parameter count. MLP-only matches MLP+attention. **[source]**
- **Learning rate.**
  - The optimal LoRA LR is about **10× the full fine-tuning LR**. The fitted multiplier is 9.8, and about 15× for very short runs under ~100 steps.
  - Because of the 1/r scaling, the optimal LR is nearly independent of rank: it varies less than 2× between ranks 4 and 512. **[source]**
  - In practice for LoRA SFT on 4B–31B models: **1e-4 to 2e-4**. Unsloth's examples use 2e-4 for short runs and suggest 2e-5 only for long runs. **[source]**
  - Our 2e-5 is at the very low end. Sweep `{5e-5, 1e-4, 2e-4}` on E4B first.
- **Alpha.**
  - TM fix `α = 32` and vary `r`.
  - The common alternative is `α = 2r` (our config: r=32, α=64).
  - rsLoRA (`α/√r`) keeps the scale stable if we change rank later.
  - Pick one convention and do not change it alongside rank.
- **Rank.**
  - With small-to-medium SFT data, LoRA matches full fine-tuning until capacity runs out. **[source]**
  - A few thousand trajectories (tens of millions of tokens) is well within r=16–64. Rank is not where the gains are.
  - Given vllm#50059, **r=16 is the safe default** until the parity check clears r=32.
- **Batch size.**
  - LoRA is *less tolerant of large batches* than full fine-tuning, and raising rank does not fix that. **[source]**
  - Keep the effective batch modest: about 8–32 sequences per optimizer step.
- **Schedule.** For a run whose length is set by a clock, use **WSD** (warmup → stable → decay):
  - linear warmup over about 3% of steps;
  - constant LR;
  - a cosine or linear decay over the last 10–20%, *triggered by the time budget*.
  - Cosine-to-a-fixed-step-count breaks when the deadline cuts the run early, because the LR never decays.
- **Dropout and weight decay.** 0–0.05 dropout; weight decay ~0. Neither matters much for LoRA.
- **Epochs.** Typically 1–3 on SFT data. Watch held-out val loss per epoch, since agent SFT overfits quickly on repeated trajectories. Prefer more unique trajectories over more epochs.

---

## 6. The loss for agent trajectories

- **Assistant-only loss.**
  - Mask system, user and tool tokens.
  - **Do** train on the assistant's tool-call tokens: they are the actions.
  - Our `processor.tokenize_trajectory` already builds this mask from the served template.
- **Normalization.**
  - Use the mean over *supervised tokens in the whole optimizer step*. That is transformers' `num_items_in_batch` path, fixed in 4.46+.
  - Do not average per micro-batch: that silently upweights short sequences under gradient accumulation.
- **Per-span weights** (e.g., 1.5× on `<|tool_call>` spans).
  - Easy to add as a `loss_weights` column with a custom `compute_loss`.
  - It is an unproven trick, so ship it **off by default** and A/B it. The cleaner lever is data: more trajectories with the behavior you want.
- **Long trajectories.**
  - *Dropping* everything over `max_seq_length` biases the model away from hard tasks, the ones that need long runs.
  - Alternative: **window** them the way the harness compacts context. The harness's `EventsCompactionConfig` summarizes at about 14,336 tokens and keeps 5 events, so training on post-compaction views teaches the model the contexts it will actually see.
  - At minimum, log the dropped fraction and the length histogram.
- **Mixing roles.**
  - With separate adapters, each adapter sees only its role's conversations.
  - With one shared adapter, include both roles. The system prompt tells the model which role it is playing.

---

## 7. Throughput

- **Batching.** With sequences of 8k–16k tokens, `per_device_train_batch_size=1` plus gradient accumulation wastes no padding. Group by length so micro-batches are similar.
- **Packing / padding-free.**
  - Concatenating short examples needs varlen attention (FlashAttention) to stop examples attending to each other.
  - That is off the table on T4, and probably for the 512-dim global layers (§2).
  - Our data is long anyway, so packing gains little.
- **Gradient checkpointing.** Use non-reentrant (`use_reentrant=False`). Unsloth's "offloaded" checkpointing trades PCIe time for memory.
- **Rough tokens per second.** **[estimate]** Assumes LoRA ≈ 6·N FLOPs per token with checkpointing and ~35% MFU, with E4B at ~4B compute parameters.

| Setup | Tokens/s | Tokens in 10 h | Trajectory-passes at ~12k tokens each |
|---|---|---|---|
| E4B, 1× L4 | ~1.3k | ~45M | ~3,800 |
| E4B, 4× L4 (DDP or 4 parallel runs) | ~5k | ~180M | ~15,000 |
| 31B QLoRA, 4× L4 model split (1 GPU busy at a time) | ~200 | ~7M | ~600 |
| 31B, FSDP-QLoRA across 4× L4 | ~600–800 | ~25M | ~2,000 |

So **data is the bottleneck for E4B, and compute is the bottleneck for 31B**. The pipeline should *measure* tokens/s in the first minutes and print the projected passes.

---

## 8. Robustness: what makes a pipeline "run it again and trust it"

The internal pipelines at Anthropic, OpenAI and similar labs are not public. Below are the shared principles behind public, production-grade fine-tuning stacks: TRL, Tinker, Axolotl, torchtune, Unsloth.

1. **Config is the experiment.** One YAML fully defines a run:
   - model, data paths, template, LoRA, optimizer, time budget, eval tasks.
   - The notebook only picks a config and overrides. The run directory stores the resolved config, git sha, data hashes, template sha256 and library versions.
2. **Preflight on CPU before spending GPU.** Fail in minute 1, not hour 7. Checks:
   - data schema;
   - leakage against the 129 test tasks;
   - template render plus tool-parser round trip (§4.4);
   - token-length histogram and supervised-token share;
   - dropped/windowed counts;
   - projected run time.
3. **OOM in minute one.** Put the *longest* batch first for a few steps, so a memory blowup happens immediately. Then shuffle.
4. **Time budget is a first-class input.** For example:
   - `train_deadline = start + 10h15m`
   - a WSD decay that begins at a set fraction of the budget
   - a hard stop that always saves
   - Kaggle kills at 12 h, and a killed run may save nothing.
5. **Checkpoint by wall clock** (e.g., every 45 minutes) to `/kaggle/working`, keeping only the last one or two. A crashed or killed session can then resume from the previous version's output in the next session.
6. **Phases as separate processes.** Order: preflight → train → export → eval → package.
   - A fresh process guarantees GPU memory is free before vLLM starts.
   - A crash in eval cannot lose the adapter.
7. **Always produce something.**
   - Every phase writes a status file.
   - Packaging runs even when eval fails.
   - The manifest says exactly which phase failed and why.
8. **Watch the right signals.** Log at least these:
   - loss (train, val, per role);
   - grad norm;
   - LR;
   - tokens/s;
   - peak GPU memory;
   - supervised tokens per step.
   Also run a **dead-adapter check**: LoRA B matrices still zero after N steps means no gradient is reaching that layer.
9. **Determinism where it is cheap.** Seed everything and keep the data order fixed. Record shuffling seeds in the manifest.

---

## 9. Evaluating inside the run

| Signal | Cost | What it tells you |
|---|---|---|
| Val loss on held-out trajectories (by task, not by turn), per role | seconds | Under- or over-fitting; LR choice |
| Teacher-forced next-action accuracy: given the prefix, is the predicted tool name correct and are the arguments parseable? | minutes | Whether the agent behavior was learned, not just tokens |
| Adapter parity: HF vs vLLM log-probs on fixed prompts (§4.3) | minutes | Whether vLLM serves what we trained |
| Live smoke eval: N tasks via `swegemma` `Evaluator`, subprocess sandbox | ~5–10 min per task, plus vLLM start-up (31B with TP=4 takes several minutes) | That the whole submission works end to end; no score |

**Why 3 tasks cannot rank models.** These are binomial probabilities for a model with true pass rate `p`:

| Observed | p = 0.3 | p = 0.5 |
|---|---|---|
| 0/3 | 34% | 12.5% |
| 1/3 | 44% | 37.5% |
| 2/3 | 19% | 37.5% |
| 3/3 | 3% | 12.5% |

- Fix the 3 tasks so runs are comparable: easy-to-medium, different repos, all from `tasks.jsonl`, never trained on.
- Report the score alongside tool-call parse errors, steps used and patch-applied rate.
- Ranking models needs the Evaluation track's larger runs.

---

## 10. SGLang vs vLLM

- **For SFT training: neither.** Training is a PyTorch forward and backward pass (TRL/PEFT). An inference server only appears when you *generate*:
  - our end-of-run eval;
  - rollouts for data collection, DPO negatives or RL.
- **For our eval: vLLM, the same version as the scorer (0.19.1, from the official wheelhouse).**
  - The score must reflect what Kaggle will run: the `gemma4` tool and reasoning parsers, LoRA on compressed-tensors (with its possible bug, §4.3), sampling defaults and context limits.
  - A different engine could hide or invent failures.
- **Where SGLang could help later:** high-throughput rollout generation with many LoRAs and shared prefixes. Benchmarks claim an edge there. **[source]**
  - That is the Data track's generator, and it still has to match the Gemma 4 tool format.
  - Adding a second engine now buys nothing for SFT.

---

## 11. Unsloth vs TRL + PEFT

| | TRL + PEFT + Accelerate | Unsloth |
|---|---|---|
| Speed / VRAM (single GPU) | baseline | ~1.5× faster, ~60% less VRAM for Gemma 4 **[source]** |
| Multi-GPU (4× L4) | DDP, FSDP (incl. FSDP-QLoRA), device_map split | DDP guide exists; FSDP not supported; reports of DDP failures **[source]** |
| Gemma 4 quirks | You handle them: template, softcap in custom loss, fp16 on T4 | Handled: its own templates (`gemma-4`, `gemma-4-thinking`), chunked CE, fp16 and KV-sharing fixes |
| Assistant-only loss for our tool trajectories | Our own `tokenize_trajectory` mask (exact, served-template based) | `train_on_responses_only` splits on `<|turn>model\n`. It would also train on text we mask, unless we pass pre-tokenized labels. |
| Offline install on L4 (no internet) | Few wheels | Many pinned wheels (`unsloth`, `unsloth_zoo`, triton, …); patches transformers internals, so version drift is fragile |
| Stability across releases | High | Fast-moving; breaking changes common |

**Recommendation:**

- Backbone: **TRL + PEFT + Accelerate**.
- Memory: fix it with supervised-position logits or Liger, gradient checkpointing and QLoRA where needed.
- Speed: Unsloth's gain is real but mostly matters on single small GPUs. Our L4 time is limited by quota, not tokens/s for E4B.
- Revisit Unsloth only if the 31B run is compute-bound *and* it supports our multi-GPU layout.

---

## 12. Multi-LoRA: one adapter or two?

- **What the harness supports** **[verified]**:
  - one base model;
  - any number of adapters under `adapters/<name>/`, selected per agent via `adapter: <name>`;
  - vLLM loads all of them (`max_loras = count`, `max_lora_rank = max rank`).
- **For two role adapters** (`main_lora` for the root coder, `tool_lora` for the read-only `code_analyzer` sub-agent):
  - Each role has a narrow job: the analyzer reads and searches and never edits; the root edits, tests and submits.
  - The two don't interfere when trained separately.
  - The analyzer's conversations are separate sessions (`agent_tool` with `skip_summarization`), so they are naturally separate training examples.
- **Against:**
  - Each adapter sees about half the data.
  - Twice the training time.
  - Twice the exposure to the vLLM LoRA bug.
  - The root model still has to learn *when* to delegate.
- **Pipeline consequence:** treat "adapters" as a list in the config, where each entry has a name, a role filter over the data and optional hyperparameter overrides. Then the experiment that decides it is:
  - `[shared]` — one adapter, both roles;
  - `[main_lora, tool_lora]` — one adapter per role.
- **The data contract this needs:** every trajectory record carries `agent` / `role` (`root` or `code_analyzer`), produced by the Data track.

---

## 13. Run economics

- About 30 h/week of GPU quota at 2× for L4 gives roughly **one full L4 run per week**, so maybe **4 full L4 runs before the 25 Nov entry deadline**.
- To make each run count:
  - Preflight locally or on CPU.
  - Use 20–30-minute **calibration runs** to measure tokens/s and memory before committing 12 hours.
  - For E4B, run **four configs in parallel** in one L4 session (one per GPU), e.g. an LR sweep, or shared vs per-role adapters.
  - Keep T4 for cheap code-path tests only: different kernels, fp16, and no vLLM Gemma 4 without a patch.

---

## 14. Things to verify on the first real runs (checklist)

- [ ] On the L4 image, which of `peft`/`trl`/`accelerate`/`datasets`/`liger-kernel` are preinstalled, and at what versions? Build the private wheelhouse for the rest, pinned against `transformers==5.13.1`.
- [ ] Which `attn_implementation` transformers 5.13.1 uses for Gemma 4 training (head dim 512), and its memory at 16k.
- [ ] Does Liger (or Cut Cross-Entropy) support Gemma 4 with the softcap? If not, implement supervised-position logits.
- [ ] Dead-adapter check on E4B's KV-shared layers (`k_proj`/`v_proj`).
- [ ] Adapter parity HF vs vLLM 0.19.1 for r=16 and r=32 (vllm#50059).
- [ ] Does the ADK/LiteLLM path send previous `reasoning` back in history? Match training rendering to the real trace.
- [ ] Does PEFT LoRA work over `run_compressed=True` compressed-tensors linears (option C, §4.2)?
- [ ] Measured tokens/s and peak memory per model and GPU layout. Replace §7's estimates.

---

## Sources

- Official starter notebook `ryanholbrook/getting-started-gemma-4-developer-agent` and its `kernel-metadata.json` (pulled 2026-10-29): `machine_shape: NvidiaL4`, wheelhouse, `VllmConfig`, `EvalConfig(sandbox='subprocess')`.
- Official wheelhouse `metric/gemma-4-developer-agent-wheelhouse` (file listing, 2026-10-29).
- `build/gemma-4-developer-agent/HARNESS_README.md` §3.4 (multi-LoRA serving), §4.1 (sandboxes), §9 (CLI).
- `DataMining/template_check/report.json`: chat-template hashes across Kaggle and HF checkpoints.
- Model configs: <https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct>, <https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct>
- Competition overview (L4×4 at 2× quota, internet disabled): <https://www.kaggle.com/competitions/gemma-4-developer-agent/overview>
- Thinking Machines, *LoRA Without Regret*: <https://thinkingmachines.ai/blog/lora/>; TRL reproduction: <https://huggingface.co/docs/trl/main/en/lora_without_regret>
- Unsloth Gemma 4 fine-tuning guide: <https://unsloth.ai/docs/models/gemma-4/train>; multi-GPU: <https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth>
- vLLM issue #50059, LoRA on compressed-tensors int4: <https://github.com/vllm-project/vllm/issues/50059>
- Gemma 4 on a Tesla T4 (Turing attention patch, W4A16 kernels): <https://dev.to/gde/gemma-4-on-a-tesla-t4-qat-weights-decode-179x-faster-than-bf16-2fi4>, <https://dev.to/gde/gemma-4-on-a-tesla-t4-part-3-int4-embeddings-serve-e2b-in-286-gib-at-230x-bf16-3kch>
- Kaggle CLI machine shapes: <https://github.com/Kaggle/kaggle-cli/issues/1196>, <https://github.com/Kaggle/kaggle-cli/pull/1198>
- TRL SFT docs (`assistant_only_loss`, `padding_free`, packing): <https://huggingface.co/docs/trl/main/en/sft_trainer>
- Liger Kernel: <https://github.com/linkedin/Liger-Kernel>
- SGLang vs vLLM comparison (third-party): <https://www.morphllm.com/comparisons/sglang-vs-vllm>
- Kaggle free-tier limits (third-party summaries): <https://gpuperhour.com/blog/free-cloud-gpus-and-credits>, <https://aifree.dev/offer/kaggle-free-gpu-tpu>
