"""Resume compatibility: what must match between a checkpoint and the run that wants to continue from it.

A checkpoint stores a *fingerprint* of the settings that define what the saved weights mean. On resume the current
settings are compared with it:

* fatal   - the adapter would silently mean something else (different model, rank, alpha, target modules, adapter).
* warning - the weights still load but part of the state is not reusable (optimizer from another backend) or the
            conditions changed (different quantization, tokenizer, template).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

FINGERPRINT_VERSION = 1


class ResumeIncompatible(ValueError):
    """The checkpoint was produced by settings that make resuming from it unsafe."""


@dataclass
class Problem:
    level: str  # "fatal" | "warn"
    code: str
    message: str


@dataclass
class ResumeVerdict:
    problems: list[Problem] = field(default_factory=list)
    optimizer_compatible: bool = True

    @property
    def fatal(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "fatal"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "warn"]


def training_fingerprint(
    *,
    adapter: str,
    served_model_id: str,
    train_base: str | None,
    quantization: str,
    backend: str,
    lora: dict[str, Any],
) -> dict[str, Any]:
    """Settings that define what the saved LoRA weights mean."""
    return {
        "version": FINGERPRINT_VERSION,
        "adapter": adapter,
        "served_model_id": served_model_id,
        "train_base": train_base,  # a directory name such as 'gemma-4-e4b-it-qat-q4_0-unquantized/2'
        "quantization": quantization,
        "backend": backend,
        "lora": {
            "r": int(lora["r"]),
            "alpha": float(lora["alpha"]),
            "target_regex": lora.get("target_regex"),
            "use_rslora": bool(lora.get("use_rslora", False)),
            "exclude_kv_shared": bool(lora.get("exclude_kv_shared", True)),
        },
    }


def example_keys_digest(keys: Iterable[str]) -> str:
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:20]


def data_fingerprint(
    keys: Iterable[str], *, tokenizer_sha: str | None = None, template_sha: str | None = None, max_seq_length: int | None = None
) -> dict[str, Any]:
    ks = list(keys)
    return {
        "count": len(ks),
        "keys_sha": example_keys_digest(ks),
        "tokenizer_sha": tokenizer_sha,
        "template_sha": template_sha,
        "max_seq_length": max_seq_length,
    }


def compare_fingerprints(saved: dict[str, Any] | None, current: dict[str, Any]) -> ResumeVerdict:
    """Compares a checkpoint's fingerprint with the current one."""
    verdict = ResumeVerdict()
    if not saved:
        verdict.problems.append(
            Problem("warn", "legacy_checkpoint", "checkpoint has no fingerprint (written by an older pipeline); compatibility unchecked")
        )
        return verdict

    def fatal(code: str, msg: str) -> None:
        verdict.problems.append(Problem("fatal", code, msg))

    def warn(code: str, msg: str) -> None:
        verdict.problems.append(Problem("warn", code, msg))

    if saved.get("adapter") != current.get("adapter"):
        fatal("adapter", f"checkpoint is for adapter {saved.get('adapter')!r}, this run trains {current.get('adapter')!r}")
    if saved.get("served_model_id") != current.get("served_model_id"):
        fatal("served_model", f"checkpoint targets {saved.get('served_model_id')!r}, this run targets {current.get('served_model_id')!r}")
    s_lora, c_lora = saved.get("lora") or {}, current.get("lora") or {}
    for key in ("r", "alpha", "target_regex", "use_rslora", "exclude_kv_shared"):
        if key in s_lora and s_lora.get(key) != c_lora.get(key):
            fatal(f"lora_{key}", f"lora.{key} changed: checkpoint {s_lora.get(key)!r} vs now {c_lora.get(key)!r}")
    if saved.get("train_base") and current.get("train_base") and saved["train_base"] != current["train_base"]:
        warn("train_base", f"base checkpoint changed: {saved['train_base']!r} -> {current['train_base']!r}")
    if saved.get("quantization") != current.get("quantization"):
        warn("quantization", f"base precision changed: {saved.get('quantization')!r} -> {current.get('quantization')!r}")
    if saved.get("backend") != current.get("backend"):
        verdict.optimizer_compatible = False
        warn("backend", f"trainer backend changed ({saved.get('backend')!r} -> {current.get('backend')!r}); adapter weights are reused, optimizer state is not")
    return verdict


def compare_data(saved: dict[str, Any] | None, current: dict[str, Any]) -> list[Problem]:
    """Data-level differences are never fatal: the ledger decides what to skip. They are reported for the log."""
    if not saved:
        return []
    out: list[Problem] = []
    if saved.get("keys_sha") != current.get("keys_sha"):
        out.append(Problem("warn", "data_changed", f"training set changed: {saved.get('count')} -> {current.get('count')} examples"))
    for key in ("tokenizer_sha", "template_sha"):
        if saved.get(key) and current.get(key) and saved[key] != current[key]:
            out.append(Problem("warn", key, f"{key} changed since the checkpoint was written"))
    return out


def enforce(verdict: ResumeVerdict, *, strict: bool = True) -> ResumeVerdict:
    """Raises `ResumeIncompatible` on fatal problems when `strict`; otherwise leaves the decision to the caller."""
    if strict and verdict.fatal:
        raise ResumeIncompatible(
            "Refusing to resume: " + "; ".join(p.message for p in verdict.fatal) + ". Start fresh (clear resume_from) or restore the old settings."
        )
    return verdict
