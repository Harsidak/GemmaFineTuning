"""Run directory layout, phase status files, code/file hashing, manifest, and logging helpers.

Follows design_doc.md §6.1 and §10.3:
  runs/<run_id>/
    resolved_config.yaml   manifest.json   REPORT.md   submission.zip
    status/<phase>.json    logs/<phase>.log            metrics/<adapter>.jsonl
    preflight/report.json  cache/<adapter>/            checkpoints/<adapter>/step_<n>/
    adapters/<name>/       parity/                     submission/
    eval/results/          eval/score.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig

TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "safetensors",
    "pyarrow",
    "pyyaml",
    "vllm",
    "compressed_tensors",
    "swegemma",
    "adk_submission",
    "adk_eval_core",
    "google_adk",
)


def utc_iso(ts: float | None = None) -> str:
    if ts is None:
        return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_pipeline_code_hash(pipeline_dir: str | Path | None = None) -> str:
    """Computes a deterministic sha256 across all `*.py` and `*.yaml` files under `pipeline/`."""
    if pipeline_dir is None:
        pipeline_dir = Path(__file__).resolve().parents[1]
    root = Path(pipeline_dir)
    h = hashlib.sha256()
    candidates: list[Path] = []
    for ext in ("*.py", "*.yaml", "*.yml"):
        candidates.extend(root.rglob(ext))
    for file_path in sorted(set(candidates), key=lambda p: p.relative_to(root).as_posix()):
        rel = file_path.relative_to(root).as_posix()
        if "__pycache__" in rel or "/runs/" in rel or rel.startswith("runs/"):
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def collect_package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in packages:
        for candidate in (pkg, pkg.replace("_", "-")):
            try:
                versions[pkg] = importlib.metadata.version(candidate)
                break
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = None
    return versions


@dataclass
class PhaseStatus:
    phase: str
    state: str  # "ok" | "failed" | "timeout" | "skipped" | "running"
    started: str
    ended: str | None = None
    started_ts: float = 0.0
    ended_ts: float | None = None
    planned_deadline: str | None = None
    planned_deadline_ts: float | None = None
    planned_duration_sec: float | None = None
    actual_duration_sec: float | None = None
    error: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunContext:
    """Manages filesystem paths, status JSONs, and manifest persistence for a single pipeline run."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = run_id or self.run_dir.name

        self.status_dir = self.run_dir / "status"
        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.preflight_dir = self.run_dir / "preflight"
        self.cache_dir = self.run_dir / "cache"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.adapters_dir = self.run_dir / "adapters"
        self.parity_dir = self.run_dir / "parity"
        self.submission_dir = self.run_dir / "submission"
        self.eval_dir = self.run_dir / "eval"
        self.eval_results_dir = self.eval_dir / "results"

        self.resolved_config_path = self.run_dir / "resolved_config.yaml"
        self.manifest_path = self.run_dir / "manifest.json"
        self.report_path = self.run_dir / "REPORT.md"
        self.submission_zip_path = self.run_dir / "submission.zip"
        self.preflight_report_path = self.preflight_dir / "report.json"
        self.eval_score_path = self.eval_dir / "score.json"

    @classmethod
    def create(cls, cfg: PipelineConfig, t0: float) -> RunContext:
        out_root = Path(cfg.run.output_root)
        if cfg.run.run_id:
            run_id = cfg.run.run_id
        else:
            date_tag = dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            run_id = f"{date_tag}_{cfg.run.name}"
        ctx = cls(out_root / run_id, run_id=run_id)
        ctx.ensure_dirs()
        return ctx

    def ensure_dirs(self) -> None:
        for d in (
            self.run_dir,
            self.status_dir,
            self.logs_dir,
            self.metrics_dir,
            self.preflight_dir,
            self.cache_dir,
            self.checkpoints_dir,
            self.adapters_dir,
            self.parity_dir,
            self.eval_dir,
            self.eval_results_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def status_filename(phase: str) -> str:
        safe = phase.replace(":", "__").replace("/", "_")
        return f"{safe}.json"

    def status_path(self, phase: str) -> Path:
        return self.status_dir / self.status_filename(phase)

    def log_path(self, phase: str) -> Path:
        safe = phase.replace(":", "__").replace("/", "_")
        return self.logs_dir / f"{safe}.log"

    def write_status(self, status: PhaseStatus) -> Path:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        p = self.status_path(status.phase)
        p.write_text(json.dumps(status.to_dict(), indent=2), encoding="utf-8")
        return p

    def read_status(self, phase: str) -> dict[str, Any] | None:
        p = self.status_path(phase)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def read_all_statuses(self) -> list[dict[str, Any]]:
        if not self.status_dir.exists():
            return []
        items = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(self.status_dir.glob("*.json"))
        ]
        items.sort(key=lambda d: d.get("started_ts", 0.0))
        return items

    def read_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def update_manifest(self, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.read_manifest()
        current.update(patch)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return current


def configure_phase_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures root/phase logging to stdout and an optional UTF-8 log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("multi_lora_sft")
