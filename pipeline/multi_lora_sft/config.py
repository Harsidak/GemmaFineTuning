"""Configuration schema, YAML loader, `--set` override parser, and strict validator.

Follows design_doc.md §3.3, §4, §10.1:
- Unknown keys anywhere in the config or `--set` overrides raise `ConfigError`.
- Per-adapter `overrides` are deep-merged onto global `lora`, `train`, `loss`, and `schedule` sections
  and validated with the same rules (e.g. rank <= 128).
- Allowed served models are strictly the four competition W4A16 quantized checkpoints.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SERVED_MODELS: frozenset[str] = frozenset(
    {
        "gemma-4-31b-it-qat-w4a16-ct",
        "gemma-4-12b-it-qat-w4a16-ct",
        "gemma-4-e4b-it-qat-w4a16-ct",
        "gemma-4-e2b-it-qat-w4a16-ct",
    }
)

ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_LORA_RANK = 128
MAX_CONTEXT_TOKENS = 32_768
MAX_SUBMISSION_BYTES = 3 * 1024 * 1024 * 1024  # 3 GiB = 3,221,225,472 bytes

DEFAULT_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class ConfigError(ValueError):
    """Raised when configuration loading, override merging, or validation fails."""


@dataclass
class RunSection:
    name: str = "multi_lora_sft"
    mode: str = "full"  # "full" | "calibration"
    calibration_minutes: float = 30.0
    seed: int = 1234
    output_root: str = "/kaggle/working/runs"
    run_id: str | None = None
    resume_from: str | None = None
    session_start_env: str = "SESSION_T0"
    git_sha: str | None = None


@dataclass
class ResumeSection:
    extend: bool = False


@dataclass
class BudgetSection:
    session_hours: float = 12.0
    safety_margin_min: float = 20.0
    eval_min: float = 60.0
    package_min: float = 5.0
    export_min: float = 10.0
    setup_max_min: float = 10.0
    preflight_max_min: float = 15.0


@dataclass
class SetupSection:
    wheelhouse_dirs: list[str] = field(default_factory=list)
    install_wheels: bool = False
    require_gpu: bool = False
    expected_gpu_count: int | None = None


@dataclass
class ModelSection:
    train_base_path: str | None = None
    served_path: str | None = None
    served_model_id: str = "gemma-4-31b-it-qat-w4a16-ct"
    expected_template_sha256: str | None = None
    quantization: str = "nf4"  # "bf16" | "nf4" | "fp32"
    attn_implementation: str = "sdpa"  # "sdpa" | "eager" | "flash_attention_2"
    max_memory_gib: float = 20.0
    skip_multimodal_towers: bool = True


@dataclass
class DataSection:
    train_files: list[str] = field(default_factory=list)
    tasks_file: str | None = "/kaggle/input/gemma-4-developer-agent/tasks.jsonl"
    successful_only: bool = True
    val_fraction: float = 0.05
    val_max_examples: int = 64
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    cache_dir: str | None = None
    num_workers: int = 0
    max_template_mismatch_fraction: float = 0.05
    repo_commit_warn_distance: int = 50


@dataclass
class LoraSection:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    bias: str = "none"
    use_rslora: bool = False
    init_lora_weights: bool = True
    target_regex: str = DEFAULT_TARGET_REGEX
    exclude_kv_shared: bool = True
    export_target_modules_mode: str = "explicit"  # "explicit" | "regex"


@dataclass
class TrainSection:
    backend: str = "trl"
    parallelism: str = "model_split"  # "ddp" | "model_split" | "single"
    num_processes: int | None = None
    lr: float = 1.0e-4
    max_epochs: float = 2.0
    max_steps: int | None = None
    max_seq_length: int = 16384
    per_device_batch: int = 1
    grad_accum: int = 16
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    longest_first_steps: int = 3
    length_bucket_size: int = 64
    checkpoint_every_min: float = 45.0
    keep_checkpoints: int = 2
    val_every_min: float = 60.0
    log_every_steps: int = 5
    dead_check_step: int = 50
    throughput_project_step: int = 10
    max_restarts: int = 2
    min_restart_min: float = 20.0
    oom_fallbacks: list[dict[str, Any]] = field(
        default_factory=lambda: [{"max_seq_length": 12288}, {"max_seq_length": 8192}]
    )


@dataclass
class ScheduleSection:
    warmup_steps: int = 20
    decay_frac: float = 0.15
    final_lr_ratio: float = 0.1
    step_time_ema_beta: float = 0.9


@dataclass
class LossSection:
    impl: str = "selective"  # "selective" | "liger"
    chunk_rows: int = 4096
    tool_call_weight: float = 1.0


@dataclass
class AdapterSpec:
    name: str = "main_lora"
    roles: list[str] | None = None
    time_share: float | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParitySection:
    enabled: bool = True
    num_prompts: int = 8
    max_prompt_tokens: int = 4096
    gen_tokens: int = 64
    max_agreement_drop: float = 0.10


@dataclass
class SubmissionSection:
    template_dir: str | None = None
    served_model_id: str | None = None
    adapter_map: dict[str, str] = field(
        default_factory=lambda: {"main_lora": "main_lora", "tool_lora": "tool_lora"}
    )
    fallback_adapters_dir: str | None = None
    working_zip_path: str | None = "/kaggle/working/submission.zip"


@dataclass
class VllmEvalSection:
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 4
    max_model_len: int = 32768
    startup_timeout: int = 1200


@dataclass
class EvalSection:
    enabled: bool = True
    task_ids: list[str] = field(default_factory=list)
    snapshots_dir: str | None = None
    graphs_dir: str | None = None
    embeddings_dir: str | None = None
    wheels_dir: str | None = None
    concurrency: int = 1
    sandbox: str = "subprocess"
    vllm: VllmEvalSection = field(default_factory=VllmEvalSection)


@dataclass
class EffectiveAdapterConfig:
    """Resolved per-adapter configuration after deep-merging `AdapterSpec.overrides`."""

    name: str
    roles: list[str] | None
    time_share: float | None
    lora: LoraSection
    train: TrainSection
    schedule: ScheduleSection
    loss: LossSection
    model_quantization: str


@dataclass
class PipelineConfig:
    run: RunSection = field(default_factory=RunSection)
    resume: ResumeSection = field(default_factory=ResumeSection)
    budget: BudgetSection = field(default_factory=BudgetSection)
    setup: SetupSection = field(default_factory=SetupSection)
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    lora: LoraSection = field(default_factory=LoraSection)
    train: TrainSection = field(default_factory=TrainSection)
    schedule: ScheduleSection = field(default_factory=ScheduleSection)
    loss: LossSection = field(default_factory=LossSection)
    adapters: list[AdapterSpec] = field(
        default_factory=lambda: [
            AdapterSpec(name="main_lora", roles=["root"]),
            AdapterSpec(name="tool_lora", roles=["code_analyzer"]),
        ]
    )
    parity: ParitySection = field(default_factory=ParitySection)
    submission: SubmissionSection = field(default_factory=SubmissionSection)
    eval: EvalSection = field(default_factory=EvalSection)

    def effective_served_model_id(self) -> str:
        return self.submission.served_model_id or self.model.served_model_id

    def resolve_adapter(self, adapter_or_name: AdapterSpec | str) -> EffectiveAdapterConfig:
        if isinstance(adapter_or_name, str):
            matches = [a for a in self.adapters if a.name == adapter_or_name]
            if not matches:
                raise ConfigError(f"Unknown adapter {adapter_or_name!r}")
            spec = matches[0]
        else:
            spec = adapter_or_name

        lora_dict = dataclasses.asdict(self.lora)
        train_dict = dataclasses.asdict(self.train)
        sched_dict = dataclasses.asdict(self.schedule)
        loss_dict = dataclasses.asdict(self.loss)
        model_quant = self.model.quantization

        overrides = copy.deepcopy(spec.overrides or {})
        allowed_override_sections = {"lora", "train", "schedule", "loss", "model"}
        unknown_top = set(overrides.keys()) - allowed_override_sections
        if unknown_top:
            raise ConfigError(
                f"Adapter {spec.name!r} overrides unknown section(s): {sorted(unknown_top)}. "
                f"Allowed: {sorted(allowed_override_sections)}"
            )

        if "lora" in overrides:
            _merge_known_dict(lora_dict, overrides["lora"], f"adapters[{spec.name}].overrides.lora")
        if "train" in overrides:
            _merge_known_dict(train_dict, overrides["train"], f"adapters[{spec.name}].overrides.train")
        if "schedule" in overrides:
            _merge_known_dict(sched_dict, overrides["schedule"], f"adapters[{spec.name}].overrides.schedule")
        if "loss" in overrides:
            _merge_known_dict(loss_dict, overrides["loss"], f"adapters[{spec.name}].overrides.loss")
        if "model" in overrides:
            m_ov = overrides["model"]
            if not isinstance(m_ov, dict) or set(m_ov.keys()) - {"quantization"}:
                raise ConfigError(
                    f"adapters[{spec.name}].overrides.model only supports 'quantization', got {m_ov!r}"
                )
            if "quantization" in m_ov:
                model_quant = str(m_ov["quantization"])

        return EffectiveAdapterConfig(
            name=spec.name,
            roles=list(spec.roles) if spec.roles is not None else None,
            time_share=spec.time_share,
            lora=_build_dataclass(LoraSection, lora_dict, f"adapters[{spec.name}].lora"),
            train=_build_dataclass(TrainSection, train_dict, f"adapters[{spec.name}].train"),
            schedule=_build_dataclass(ScheduleSection, sched_dict, f"adapters[{spec.name}].schedule"),
            loss=_build_dataclass(LossSection, loss_dict, f"adapters[{spec.name}].loss"),
            model_quantization=model_quant,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return out


# Free-form mapping paths whose inner keys are user-defined rather than fixed schema fields.
_FREEFORM_DICT_PATHS: frozenset[str] = frozenset(
    {
        "data.template_kwargs",
        "submission.adapter_map",
    }
)


def _merge_known_dict(base: dict[str, Any], patch: Any, path_prefix: str) -> dict[str, Any]:
    """Deep-merges `patch` onto `base`, raising `ConfigError` on any unknown key."""
    if not isinstance(patch, dict):
        raise ConfigError(f"Expected mapping at {path_prefix!r}, got {type(patch).__name__}")
    for key, val in patch.items():
        full_key = f"{path_prefix}.{key}" if path_prefix else str(key)
        if key not in base:
            raise ConfigError(f"Unknown configuration key: {full_key!r}")
        if (
            isinstance(base[key], dict)
            and isinstance(val, dict)
            and full_key not in _FREEFORM_DICT_PATHS
        ):
            _merge_known_dict(base[key], val, full_key)
        else:
            base[key] = copy.deepcopy(val)
    return base


def parse_set_override(expr: str) -> tuple[str, Any]:
    """Parses `a.b.c=value` where `value` is interpreted via `yaml.safe_load`."""
    key, sep, raw_val = expr.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"Invalid --set override {expr!r}; expected KEY=VALUE")
    try:
        parsed = yaml.safe_load(raw_val) if raw_val != "" else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML value in --set {expr!r}: {exc}") from exc
    return key, parsed


def apply_dotted_override(raw_cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Applies a dotted path override (`a.b.c=val`) onto `raw_cfg`, checking schema validity."""
    parts = dotted_key.split(".")
    cur: Any = raw_cfg
    traversed: list[str] = []

    for idx, part in enumerate(parts[:-1]):
        traversed.append(part)
        prefix = ".".join(traversed)
        if isinstance(cur, dict):
            parent_prefix = ".".join(traversed[:-1])
            if parent_prefix not in _FREEFORM_DICT_PATHS and part not in cur:
                raise ConfigError(f"Unknown configuration key in override: {dotted_key!r} (at {prefix!r})")
            if part not in cur or cur[part] is None:
                cur[part] = {}
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            i = int(part)
            if i < 0 or i >= len(cur):
                raise ConfigError(f"List index out of range in override {dotted_key!r}: {i}")
            cur = cur[i]
        else:
            raise ConfigError(f"Cannot traverse into {prefix!r} for override {dotted_key!r}")

    leaf = parts[-1]
    parent_path = ".".join(parts[:-1])
    if isinstance(cur, dict):
        # Allow free-form dict paths and adapter overrides dictionaries (validated later in validate_config)
        is_adapter_override = len(parts) >= 3 and parts[0] == "adapters" and "overrides" in parts
        if parent_path not in _FREEFORM_DICT_PATHS and not is_adapter_override and leaf not in cur:
            raise ConfigError(f"Unknown configuration key in override: {dotted_key!r}")
        if (
            leaf in cur
            and isinstance(cur[leaf], dict)
            and isinstance(value, dict)
            and f"{parent_path}.{leaf}".strip(".") not in _FREEFORM_DICT_PATHS
            and not is_adapter_override
        ):
            _merge_known_dict(cur[leaf], value, f"{parent_path}.{leaf}".strip("."))
        else:
            cur[leaf] = copy.deepcopy(value)
    elif isinstance(cur, list) and leaf.isdigit():
        i = int(leaf)
        if i < 0 or i >= len(cur):
            raise ConfigError(f"List index out of range in override {dotted_key!r}: {i}")
        cur[i] = copy.deepcopy(value)
    else:
        raise ConfigError(f"Cannot set {leaf!r} on {parent_path!r} in override {dotted_key!r}")


def _coerce_scalar(val: Any, target_name: str, field_path: str) -> Any:
    if val is None:
        return None
    if target_name == "float" and isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if target_name == "int" and isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    if target_name == "bool":
        if not isinstance(val, bool):
            raise ConfigError(f"Expected boolean at {field_path!r}, got {val!r}")
        return val
    if target_name == "str":
        if not isinstance(val, str):
            raise ConfigError(f"Expected string at {field_path!r}, got {val!r}")
        return val
    return val


def _build_dataclass(cls: type, data: dict[str, Any], prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping for {prefix or cls.__name__}, got {type(data).__name__}")
    known_fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data.keys()) - set(known_fields.keys())
    if unknown:
        unk_list = ", ".join(f"{prefix}.{k}".strip(".") for k in sorted(unknown))
        raise ConfigError(f"Unknown configuration key(s): {unk_list}")

    kwargs: dict[str, Any] = {}
    for name, f_info in known_fields.items():
        if name not in data:
            continue
        val = data[name]
        f_path = f"{prefix}.{name}".strip(".")
        t_str = str(f_info.type)
        if name == "vllm" and cls is EvalSection:
            kwargs[name] = _build_dataclass(VllmEvalSection, val or {}, f_path)
        elif name == "adapters" and cls is PipelineConfig:
            if not isinstance(val, list):
                raise ConfigError(f"'adapters' must be a list, got {type(val).__name__}")
            kwargs[name] = [
                _build_dataclass(AdapterSpec, item, f"adapters[{idx}]")
                for idx, item in enumerate(val)
            ]
        elif t_str in {"float", "float | None"}:
            if val is None and "None" not in t_str:
                raise ConfigError(f"{f_path!r} cannot be null")
            if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float))):
                raise ConfigError(f"{f_path!r} must be a number, got {val!r}")
            kwargs[name] = _coerce_scalar(val, "float", f_path)
        elif t_str in {"int", "int | None"}:
            if val is None and "None" not in t_str:
                raise ConfigError(f"{f_path!r} cannot be null")
            if val is not None and (isinstance(val, bool) or not isinstance(val, int)):
                raise ConfigError(f"{f_path!r} must be an integer, got {val!r}")
            kwargs[name] = _coerce_scalar(val, "int", f_path)
        elif t_str == "bool":
            kwargs[name] = _coerce_scalar(val, "bool", f_path)
        elif t_str in {"str", "str | None"}:
            if val is None and "None" not in t_str:
                raise ConfigError(f"{f_path!r} cannot be null")
            kwargs[name] = _coerce_scalar(val, "str", f_path)
        else:
            kwargs[name] = copy.deepcopy(val)
    return cls(**kwargs)


def _validate_lora_section(lora: LoraSection, label: str) -> None:
    if not (1 <= lora.r <= MAX_LORA_RANK):
        raise ConfigError(f"{label}.r must be in 1..{MAX_LORA_RANK}, got {lora.r}")
    if lora.alpha <= 0:
        raise ConfigError(f"{label}.alpha must be > 0, got {lora.alpha}")
    if not (0.0 <= lora.dropout < 1.0):
        raise ConfigError(f"{label}.dropout must be in [0.0, 1.0), got {lora.dropout}")
    if lora.bias not in {"none", "all", "lora_only"}:
        raise ConfigError(f"{label}.bias must be 'none', 'all', or 'lora_only', got {lora.bias!r}")
    if lora.export_target_modules_mode not in {"explicit", "regex"}:
        raise ConfigError(
            f"{label}.export_target_modules_mode must be 'explicit' or 'regex', "
            f"got {lora.export_target_modules_mode!r}"
        )
    try:
        pat = re.compile(lora.target_regex)
    except re.error as exc:
        raise ConfigError(f"{label}.target_regex is not a valid regex: {exc}") from exc

    # Guard against accidentally matching multimodal towers (design_doc.md §3.4, §11).
    forbidden_probes = [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
        "model.audio_tower.layers.0.self_attn.k_proj",
    ]
    for probe in forbidden_probes:
        if pat.fullmatch(probe) or (not lora.target_regex.startswith(".*") and pat.search(probe)):
            raise ConfigError(
                f"{label}.target_regex {lora.target_regex!r} matches multimodal module {probe!r}; "
                "pin the regex to 'language_model'."
            )


def _validate_train_section(train: TrainSection, label: str) -> None:
    if train.backend not in {"trl"}:
        raise ConfigError(f"{label}.backend {train.backend!r} is not supported in v1 (expected 'trl')")
    if train.parallelism not in {"ddp", "model_split", "single"}:
        raise ConfigError(
            f"{label}.parallelism must be 'ddp', 'model_split', or 'single', got {train.parallelism!r}"
        )
    if train.lr <= 0:
        raise ConfigError(f"{label}.lr must be > 0, got {train.lr}")
    if train.max_epochs <= 0:
        raise ConfigError(f"{label}.max_epochs must be > 0, got {train.max_epochs}")
    if train.max_steps is not None and train.max_steps <= 0:
        raise ConfigError(f"{label}.max_steps must be > 0 when set, got {train.max_steps}")
    if not (1 <= train.max_seq_length <= MAX_CONTEXT_TOKENS):
        raise ConfigError(
            f"{label}.max_seq_length must be in 1..{MAX_CONTEXT_TOKENS}, got {train.max_seq_length}"
        )
    if train.per_device_batch < 1 or train.grad_accum < 1:
        raise ConfigError(f"{label}.per_device_batch and grad_accum must be >= 1")
    if train.keep_checkpoints < 1:
        raise ConfigError(f"{label}.keep_checkpoints must be >= 1")
    if train.max_restarts < 0:
        raise ConfigError(f"{label}.max_restarts must be >= 0")

    allowed_oom_keys = {"quantization", "max_seq_length", "per_device_batch", "grad_accum"}
    for idx, fb in enumerate(train.oom_fallbacks):
        if not isinstance(fb, dict) or not fb:
            raise ConfigError(f"{label}.oom_fallbacks[{idx}] must be a non-empty mapping")
        unk = set(fb.keys()) - allowed_oom_keys
        if unk:
            raise ConfigError(
                f"{label}.oom_fallbacks[{idx}] contains unsupported key(s) {sorted(unk)}; "
                f"allowed: {sorted(allowed_oom_keys)}"
            )


def _validate_schedule_section(sched: ScheduleSection, label: str) -> None:
    if sched.warmup_steps < 0:
        raise ConfigError(f"{label}.warmup_steps must be >= 0")
    if not (0.0 <= sched.decay_frac < 1.0):
        raise ConfigError(f"{label}.decay_frac must be in [0.0, 1.0), got {sched.decay_frac}")
    if not (0.0 <= sched.final_lr_ratio <= 1.0):
        raise ConfigError(f"{label}.final_lr_ratio must be in [0.0, 1.0], got {sched.final_lr_ratio}")
    if not (0.0 < sched.step_time_ema_beta < 1.0):
        raise ConfigError(f"{label}.step_time_ema_beta must be in (0.0, 1.0)")


def _validate_loss_section(loss: LossSection, label: str) -> None:
    if loss.impl not in {"selective", "liger"}:
        raise ConfigError(f"{label}.impl must be 'selective' or 'liger', got {loss.impl!r}")
    if loss.chunk_rows < 1:
        raise ConfigError(f"{label}.chunk_rows must be >= 1, got {loss.chunk_rows}")
    if loss.tool_call_weight <= 0.0:
        raise ConfigError(f"{label}.tool_call_weight must be > 0, got {loss.tool_call_weight}")


def validate_config(cfg: PipelineConfig) -> None:
    """Runs full structural and domain validation on a `PipelineConfig`."""
    if cfg.run.mode not in {"full", "calibration"}:
        raise ConfigError(f"run.mode must be 'full' or 'calibration', got {cfg.run.mode!r}")
    if cfg.run.calibration_minutes <= 0:
        raise ConfigError(f"run.calibration_minutes must be > 0, got {cfg.run.calibration_minutes}")

    # Budget validation
    for fname in (
        "session_hours",
        "safety_margin_min",
        "eval_min",
        "package_min",
        "export_min",
        "setup_max_min",
        "preflight_max_min",
    ):
        val = getattr(cfg.budget, fname)
        if val < 0 or (fname == "session_hours" and val <= 0):
            raise ConfigError(f"budget.{fname} must be positive, got {val}")

    if cfg.run.mode == "full":
        eval_m = cfg.budget.eval_min if cfg.eval.enabled else 0.0
        reserved_min = cfg.budget.safety_margin_min + eval_m + cfg.budget.package_min + cfg.budget.export_min
        if cfg.budget.session_hours * 60.0 <= reserved_min:
            raise ConfigError(
                f"budget.session_hours ({cfg.budget.session_hours * 60:.1f} min) leaves no time for training "
                f"after reserved phases ({reserved_min:.1f} min)"
            )

    # Served model validation (competition rule §3.3)
    served_id = cfg.effective_served_model_id()
    if served_id not in ALLOWED_SERVED_MODELS:
        raise ConfigError(
            f"Invalid served model {served_id!r}. Allowed competition models: {sorted(ALLOWED_SERVED_MODELS)}"
        )
    if (
        cfg.submission.served_model_id is not None
        and cfg.model.served_model_id != cfg.submission.served_model_id
    ):
        raise ConfigError(
            f"Conflicting model.served_model_id ({cfg.model.served_model_id!r}) and "
            f"submission.served_model_id ({cfg.submission.served_model_id!r})"
        )

    if cfg.model.quantization not in {"bf16", "nf4", "fp32"}:
        raise ConfigError(
            f"model.quantization must be 'bf16', 'nf4', or 'fp32', got {cfg.model.quantization!r}"
        )
    if cfg.model.attn_implementation not in {"sdpa", "eager", "flash_attention_2"}:
        raise ConfigError(
            f"model.attn_implementation must be 'sdpa', 'eager', or 'flash_attention_2', "
            f"got {cfg.model.attn_implementation!r}"
        )

    # Data validation
    if not isinstance(cfg.data.train_files, list) or not all(
        isinstance(p, str) and p.strip() for p in cfg.data.train_files
    ):
        raise ConfigError("data.train_files must be a list of non-empty path strings")
    if not (0.0 <= cfg.data.val_fraction < 1.0):
        raise ConfigError(f"data.val_fraction must be in [0.0, 1.0), got {cfg.data.val_fraction}")
    if cfg.data.val_max_examples < 0:
        raise ConfigError("data.val_max_examples must be >= 0")

    # Global sections
    _validate_lora_section(cfg.lora, "lora")
    _validate_train_section(cfg.train, "train")
    _validate_schedule_section(cfg.schedule, "schedule")
    _validate_loss_section(cfg.loss, "loss")

    # Adapters validation
    if not cfg.adapters:
        raise ConfigError("adapters list must contain at least one adapter")
    seen_names: set[str] = set()
    share_sum = 0.0
    for spec in cfg.adapters:
        if not ADAPTER_NAME_RE.match(spec.name):
            raise ConfigError(
                f"Invalid adapter name {spec.name!r}: must match {ADAPTER_NAME_RE.pattern}"
            )
        if spec.name in seen_names:
            raise ConfigError(f"Duplicate adapter name {spec.name!r}")
        seen_names.add(spec.name)
        if spec.roles is not None:
            if not isinstance(spec.roles, list) or not all(isinstance(r, str) and r for r in spec.roles):
                raise ConfigError(f"adapters[{spec.name}].roles must be null or a list of non-empty strings")
        if spec.time_share is not None:
            if spec.time_share <= 0.0 or spec.time_share > 1.0:
                raise ConfigError(f"adapters[{spec.name}].time_share must be in (0.0, 1.0]")
            share_sum += spec.time_share

        eff = cfg.resolve_adapter(spec)
        _validate_lora_section(eff.lora, f"adapters[{spec.name}].lora")
        _validate_train_section(eff.train, f"adapters[{spec.name}].train")
        _validate_schedule_section(eff.schedule, f"adapters[{spec.name}].schedule")
        _validate_loss_section(eff.loss, f"adapters[{spec.name}].loss")
        if eff.model_quantization not in {"bf16", "nf4", "fp32"}:
            raise ConfigError(
                f"adapters[{spec.name}].model.quantization invalid: {eff.model_quantization!r}"
            )

    if share_sum > 1.0 + 1e-6:
        raise ConfigError(f"Sum of explicit adapter time_share values exceeds 1.0 ({share_sum:.4f})")

    # Submission adapter_map validation
    if not isinstance(cfg.submission.adapter_map, dict):
        raise ConfigError("submission.adapter_map must be a mapping")
    for slot_name, target_name in cfg.submission.adapter_map.items():
        if not isinstance(slot_name, str) or not ADAPTER_NAME_RE.match(slot_name):
            raise ConfigError(f"Invalid submission.adapter_map key {slot_name!r}")
        if not isinstance(target_name, str) or not ADAPTER_NAME_RE.match(target_name):
            raise ConfigError(f"Invalid submission.adapter_map target {target_name!r}")

    # Eval validation
    if cfg.eval.vllm.max_model_len > MAX_CONTEXT_TOKENS or cfg.eval.vllm.max_model_len < 512:
        raise ConfigError(
            f"eval.vllm.max_model_len must be in 512..{MAX_CONTEXT_TOKENS}, got {cfg.eval.vllm.max_model_len}"
        )
    if not (0.0 < cfg.eval.vllm.gpu_memory_utilization <= 1.0):
        raise ConfigError("eval.vllm.gpu_memory_utilization must be in (0.0, 1.0]")
    if cfg.eval.sandbox not in {"subprocess", "docker"}:
        raise ConfigError(f"eval.sandbox must be 'subprocess' or 'docker', got {cfg.eval.sandbox!r}")


def load_config(
    config_path: str | Path | None = None,
    set_overrides: list[str] | None = None,
) -> PipelineConfig:
    """Loads a YAML config merged onto defaults, applies `--set` overrides, and validates."""
    raw = PipelineConfig().to_dict()

    if config_path is not None:
        cpath = Path(config_path)
        if not cpath.exists():
            raise ConfigError(f"Config file not found: {cpath}")
        loaded = yaml.safe_load(cpath.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Top-level YAML in {cpath} must be a mapping, got {type(loaded).__name__}")
        _merge_known_dict(raw, loaded, "")

    for expr in set_overrides or []:
        key, val = parse_set_override(expr)
        apply_dotted_override(raw, key, val)

    cfg = _build_dataclass(PipelineConfig, raw, "")
    validate_config(cfg)
    return cfg
