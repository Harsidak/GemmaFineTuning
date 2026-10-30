"""Base model loading, tiny CPU Gemma 4 fixture builder, and LoRA regex attachment.

Implements design_doc.md §3.4, §7.4, and §11:
- Loads `Gemma4ForConditionalGeneration` in bf16 (or fp32 on CPU / NF4 QLoRA via bitsandbytes).
- Optionally skips/frees vision and audio towers for text-only SFT while keeping the exact
  `Gemma4ForConditionalGeneration` hierarchy (`base_model.model.model.language_model.layers.{i}...`).
- Pins LoRA target modules to `language_model` via regex and excludes `k_proj`/`v_proj` on
  KV-shared layers (`text_config.num_kv_shared_layers`).
- Provides `save_tiny_gemma4_checkpoint()` for fast, offline CPU unit and calibration tests.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..config import EffectiveAdapterConfig, LoraSection, PipelineConfig
from .device import empty_cache, resolve_accelerator

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised with an actionable message when a base or served checkpoint cannot be used."""


_WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin")


def validate_model_dir(path: str | Path, label: str, *, for_training: bool) -> dict[str, Any]:
    """Checks a local HF checkpoint before any expensive load and returns its parsed `config.json`.

    Catches the failure modes that otherwise surface minutes into a run as an opaque traceback:
    a wrong/partial Kaggle mount, a missing or truncated weight shard, a non-Gemma-4 checkpoint, and
    pointing training at the served compressed-tensors (`-w4a16-ct`) checkpoint, which cannot be fine-tuned.
    """
    p = Path(path)
    if not p.exists():
        parent = p.parent
        siblings = sorted(x.name for x in parent.iterdir())[:20] if parent.is_dir() else []
        raise ModelLoadError(f"{label} does not exist: {p}" + (f" (parent contains: {siblings})" if siblings else ""))
    if not p.is_dir():
        raise ModelLoadError(f"{label} must be a checkpoint directory, got a file: {p}")

    cfg_file = p / "config.json"
    if not cfg_file.is_file():
        found = sorted(x.name for x in p.iterdir())[:20]
        raise ModelLoadError(
            f"{label} has no config.json: {p} (contains {found}). Point it at the directory holding "
            "config.json, e.g. .../transformers/<variant>/<version>"
        )
    try:
        hf_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"{label}/config.json is unreadable or invalid JSON: {exc}") from exc

    model_type = str(hf_cfg.get("model_type", ""))
    if not model_type.startswith("gemma4"):
        logger.warning("%s model_type is %r (expected gemma4*); continuing", label, model_type)

    if not for_training:
        return hf_cfg

    quant = hf_cfg.get("quantization_config") or {}
    quant_method = str(quant.get("quant_method", "")).lower()
    if quant_method in {"compressed-tensors", "compressed_tensors", "gptq", "awq"}:
        raise ModelLoadError(
            f"{label} is a {quant_method} quantized serving checkpoint ({p}); LoRA training needs the unquantized "
            "base (e.g. gemma-4-*-it-qat-q4_0-unquantized). Keep the -w4a16-ct checkpoint for model.served_path."
        )

    index_files = sorted(p.glob("*.safetensors.index.json")) + sorted(p.glob("pytorch_model*.bin.index.json"))
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"{label} weight index {index_files[0].name} is unreadable: {exc}") from exc
        shards = sorted(set((index.get("weight_map") or {}).values()))
        missing = [s for s in shards if not (p / s).is_file()]
        empty = [s for s in shards if (p / s).is_file() and (p / s).stat().st_size == 0]
        if missing or empty:
            raise ModelLoadError(
                f"{label} is incomplete: {len(missing)} missing and {len(empty)} empty shard(s) of {len(shards)} "
                f"(first: {(missing or empty)[:3]}). The Kaggle model mount or download is partial."
            )
    elif not any(f.suffix in _WEIGHT_SUFFIXES and f.stat().st_size > 0 for f in p.iterdir() if f.is_file()):
        raise ModelLoadError(f"{label} contains no non-empty weight files (*.safetensors / *.bin): {p}")

    return hf_cfg


def _text_dims(hf_cfg: dict[str, Any]) -> dict[str, Any]:
    text = hf_cfg.get("text_config") or hf_cfg
    return {k: text.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")}


def check_train_served_compat(train_cfg: dict[str, Any], served_cfg: dict[str, Any]) -> None:
    """Adapters trained on one Gemma 4 size cannot load on another; fail before hours of training."""
    t_dims, s_dims = _text_dims(train_cfg), _text_dims(served_cfg)
    mismatched = {
        k: (t_dims[k], s_dims[k])
        for k in t_dims
        if t_dims[k] is not None and s_dims[k] is not None and t_dims[k] != s_dims[k]
    }
    if mismatched:
        raise ModelLoadError(
            "model.train_base_path and model.served_path are different model sizes "
            f"({ {k: f'train={a}, served={b}' for k, (a, b) in mismatched.items()} }); "
            "LoRA adapters trained on one will fail to load on the other at scoring time."
        )

GEMMA4_SPECIAL_TOKENS: list[str] = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|channel>",
    "<channel|>",
    '<|"|>',
    "<|image|>",
    "<|audio|>",
    "<|video|>",
]


def canonical_gemma4_chat_template() -> str:
    """Returns TRL's canonical `gemma4_v5.jinja` chat template (equals served `w4a16-ct` E4B/E2B)."""
    try:
        import trl

        tpl_path = Path(trl.__file__).resolve().parent / "chat_templates" / "gemma4_v5.jinja"
        if tpl_path.exists():
            return tpl_path.read_text(encoding="utf-8")
    except (ImportError, OSError) as exc:
        logger.debug("Could not read trl gemma4_v5.jinja: %s", exc)
    raise FileNotFoundError("Could not locate canonical gemma4_v5.jinja in installed trl package")


def build_tiny_gemma4_tokenizer(vocab_size: int = 512) -> Any:
    """Builds an offline fast byte-level BPE tokenizer with all Gemma 4 special tokens and template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    base_alphabet = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
    vocab: dict[str, int] = {}
    for tok in GEMMA4_SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
    for ch in base_alphabet:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    # Add common multi-char merges so sequences stay compact while remaining 100% lossless
    common_words = [
        "the", "def", "return", "import", "from", "class", "self", "True", "False", "None",
        "path", "file", "config", "read", "edit", "pytest", "status", "ok", "error", "call",
        "response", "thought", "model", "user", "system", "function", "description", "type",
        "STRING", "INTEGER", "BOOLEAN", "OBJECT", "ARRAY", "properties", "required", "command",
    ]
    merges: list[tuple[str, str]] = []
    for word in common_words:
        cur = word[0]
        for ch in word[1:]:
            if len(vocab) >= vocab_size:
                break
            merged = cur + ch
            if merged not in vocab:
                vocab[merged] = len(vocab)
                merges.append((cur, ch))
            cur = merged

    while len(vocab) < vocab_size:
        filler = f"<extra_{len(vocab)}>"
        vocab[filler] = len(vocab)

    tok_obj = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tok_obj.decoder = decoders.Fuse()

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=[t for t in GEMMA4_SPECIAL_TOKENS if t not in {"<pad>", "<eos>", "<bos>", "<unk>"}],
        clean_up_tokenization_spaces=False,
    )
    fast_tok.chat_template = canonical_gemma4_chat_template()
    return fast_tok


def build_tiny_gemma4_config(
    *,
    vocab_size: int = 512,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 4,
    num_kv_shared_layers: int = 2,
    include_vision: bool = True,
    softcap: float = 30.0,
) -> Any:
    """Creates a tiny `Gemma4Config` exercising sliding+global layers, softcap, KV sharing, and vision."""
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

    # Ensure both sliding_attention and full_attention exist BEFORE `first_kv_shared` so shared KV cache is populated
    layer_types = [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(num_hidden_layers)
    ]
    layer_types[-1] = "full_attention"

    tc = Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        sliding_window=16,
        final_logit_softcapping=softcap,
        vocab_size_per_layer_input=vocab_size,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=num_kv_shared_layers,
        max_position_embeddings=4096,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        layer_types=layer_types,
    )
    vc = (
        Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        if include_vision
        else None
    )
    return Gemma4Config(
        text_config=tc,
        vision_config=vc,
        audio_config=None,
        boi_token_id=vocab_size - 4,
        eoi_token_id=vocab_size - 3,
        image_token_id=vocab_size - 2,
        video_token_id=vocab_size - 1,
        boa_token_id=vocab_size - 5,
        eoa_token_index=vocab_size - 6,
        audio_token_id=vocab_size - 7,
    )


def save_tiny_gemma4_checkpoint(
    target_dir: str | Path,
    *,
    vocab_size: int = 512,
    num_hidden_layers: int = 4,
    num_kv_shared_layers: int = 2,
    include_vision: bool = True,
    seed: int = 0,
) -> Path:
    """Materializes a tiny Gemma 4 checkpoint + fast tokenizer + `chat_template.jinja` on disk."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and (out / "tokenizer.json").exists():
        return out

    torch.manual_seed(seed)
    cfg = build_tiny_gemma4_config(
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        num_kv_shared_layers=num_kv_shared_layers,
        include_vision=include_vision,
    )
    model = Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16)
    model.save_pretrained(out)

    tok = build_tiny_gemma4_tokenizer(vocab_size=vocab_size)
    tok.save_pretrained(out)
    (out / "chat_template.jinja").write_text(tok.chat_template, encoding="utf-8")
    return out


def get_kv_shared_layer_indices(config: Any) -> list[int]:
    """Returns the 0-indexed text layer indices that reuse earlier layers' K/V states."""
    text_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    num_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0) or 0)
    if num_shared <= 0 or num_layers <= 0:
        return []
    first_shared = max(0, num_layers - num_shared)
    return list(range(first_shared, num_layers))


def build_lora_target_regex(config: Any, lora_cfg: LoraSection) -> str:
    """Builds the PEFT `target_modules` regex pinned to `language_model`, excluding KV-shared K/V."""
    base_regex = lora_cfg.target_regex
    if not lora_cfg.exclude_kv_shared:
        return base_regex

    shared_layers = get_kv_shared_layer_indices(config)
    if not shared_layers:
        return base_regex

    layer_alt = "|".join(str(idx) for idx in shared_layers)
    return rf"(?!.*language_model\.layers\.(?:{layer_alt})\.self_attn\.(?:k_proj|v_proj)$){base_regex}"


def find_matching_lora_modules(model: Any, target_regex: str) -> list[str]:
    """Returns sorted module paths in `model` matched by `target_regex` (verified linear-like)."""
    from torch import nn

    pat = re.compile(target_regex)
    matched: list[str] = []
    for name, mod in model.named_modules():
        if not name:
            continue
        # Match nn.Linear and bitsandbytes Linear4bit / Linear8bitLt
        is_linear = isinstance(mod, nn.Linear) or mod.__class__.__name__.startswith("Linear")
        if is_linear and pat.fullmatch(name):
            matched.append(name)
    return sorted(matched)


def load_base_model_for_training(
    cfg: PipelineConfig,
    eff_adapter: EffectiveAdapterConfig,
    *,
    tiny_fallback_dir: Path | None = None,
) -> Any:
    """Loads `Gemma4ForConditionalGeneration` configured for LoRA/QLoRA SFT training."""
    import torch
    from transformers import (
        AutoConfig,
        BitsAndBytesConfig,
        Gemma4ForConditionalGeneration,
    )

    model_path = cfg.model.train_base_path
    if model_path in {None, "__tiny__"}:
        if tiny_fallback_dir is None:
            raise ModelLoadError("model.train_base_path is not set")
        model_path = str(save_tiny_gemma4_checkpoint(tiny_fallback_dir))
    else:
        validate_model_dir(model_path, "model.train_base_path", for_training=True)

    accelerator = resolve_accelerator(eff_adapter.train.accelerator)
    quant = eff_adapter.model_quantization
    has_cuda = accelerator == "cuda"
    torch_dtype = torch.float32 if quant == "fp32" else torch.bfloat16

    quant_config = None
    if quant == "nf4":
        if not has_cuda:
            logger.warning("quantization='nf4' needs CUDA + bitsandbytes; falling back to bfloat16 on %s", accelerator)
            quant = "bf16"
        else:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ModelLoadError(
                    "quantization='nf4' requires bitsandbytes, which is not installed; add it to the wheelhouse "
                    "or use model.quantization=bf16"
                ) from exc
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "vision_tower", "audio_tower", "embed_vision", "embed_audio"],
            )

    local_only = Path(model_path).is_dir()
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    if cfg.model.skip_multimodal_towers:
        if hasattr(hf_cfg, "vision_config"):
            hf_cfg.vision_config = None
        if hasattr(hf_cfg, "audio_config"):
            hf_cfg.audio_config = None

    attn_impl = cfg.model.attn_implementation
    if attn_impl == "flash_attention_2" and not has_cuda:
        logger.warning("flash_attention_2 is CUDA-only; using sdpa on %s", accelerator)
        attn_impl = "sdpa"

    load_kwargs: dict[str, Any] = {
        "config": hf_cfg,
        "dtype": torch_dtype,
        "attn_implementation": attn_impl,
        "local_files_only": local_only,
        "low_cpu_mem_usage": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    parallelism = eff_adapter.train.parallelism
    if has_cuda:
        if parallelism == "model_split":
            n_gpus = torch.cuda.device_count()
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {
                i: f"{int(cfg.model.max_memory_gib)}GiB" for i in range(n_gpus)
            }
        elif parallelism == "ddp":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            load_kwargs["device_map"] = {"": local_rank}
        else:
            load_kwargs["device_map"] = {"": 0}
    # TPU/CPU: load on host; the Trainer moves (TPU single) or shards (TPU spmd) the model onto XLA devices.

    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except Exception as exc:
        # Keep the original text (an OOM stays recognisable to the orchestrator's fallback logic).
        raise ModelLoadError(
            f"Failed to load base model from {model_path} (quantization={quant}, accelerator={accelerator}, "
            f"parallelism={parallelism}): {type(exc).__name__}: {exc}"
        ) from exc

    _assert_fully_materialised(model, parallelism)
    model.config.use_cache = False
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = False

    # Freeze any remaining multimodal modules if skip_multimodal_towers was False
    for tower_attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        sub = getattr(getattr(model, "model", None), tower_attr, None)
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad_(False)

    if quant == "nf4":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif parallelism == "spmd":
        # XLA FSDPv2 applies its own per-layer checkpointing (`xla_fsdp_grad_ckpt`); HF checkpointing on top
        # would add a redundant all-gather in backward.
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    gc.collect()
    empty_cache(accelerator)

    return model


def _assert_fully_materialised(model: Any, parallelism: str) -> None:
    """Fails fast when `device_map="auto"` silently offloaded layers or weights stayed on `meta`.

    Offload makes QLoRA training 10-100x slower (or crash in backward), so it is reported as an
    out-of-memory condition to trigger the orchestrator's OOM fallback chain instead.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = sorted({k for k, v in device_map.items() if str(v) in {"cpu", "disk"}})
    if offloaded and parallelism == "model_split":
        raise ModelLoadError(
            f"Insufficient accelerator memory (out of memory): device_map offloaded {len(offloaded)} module(s) to "
            f"cpu/disk (first: {offloaded[:3]}). Lower model.max_memory_gib headroom, use nf4, or shorten "
            "train.max_seq_length."
        )
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        raise ModelLoadError(
            f"{len(meta)} parameter(s) were never loaded (still on 'meta', first: {meta[:3]}); "
            "the checkpoint does not match the model class or is missing shards."
        )


def attach_lora_adapter(model: Any, lora_cfg: LoraSection) -> tuple[Any, str, list[str]]:
    """Attaches a PEFT LoRA adapter to `model` using the KV-shared-aware language_model regex."""
    from peft import LoraConfig, TaskType, get_peft_model

    target_regex = build_lora_target_regex(model.config, lora_cfg)
    matched_modules = find_matching_lora_modules(model, target_regex)
    if not matched_modules:
        raise ValueError(f"LoRA target_regex matched 0 modules on {type(model).__name__}: {target_regex!r}")

    for m_name in matched_modules:
        if "vision_tower" in m_name or "audio_tower" in m_name or "language_model" not in m_name:
            raise ValueError(f"Unsafe LoRA target matched outside language_model: {m_name!r}")

    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        use_rslora=lora_cfg.use_rslora,
        init_lora_weights=lora_cfg.init_lora_weights,
        target_modules=target_regex,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_cfg)
    return peft_model, target_regex, matched_modules
