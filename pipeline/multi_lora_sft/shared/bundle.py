"""Cross-session checkpoint bundle: the output of one Kaggle session becomes the input of the next.

    session N   writes  /kaggle/working/lora_checkpoints/    (this module: `sync_bundle`)
    you         upload it as a (private) Kaggle dataset version
    session N+1 attaches that dataset; `discover_bundles` finds it and `resume_candidates` resumes from it

Bundle layout (every directory is a normal checkpoint root, so the trainers need no special casing):

    bundle.json                 schema, session counter, served model, per-adapter summary, git sha
    dataset-metadata.json       lets `kaggle datasets version -p <bundle>` work as-is
    <adapter>/step_NNNNNN/      newest resumable checkpoint (weights, optimizer, rng, ledger)
    <adapter>/best_adapter/     lowest validation loss so far
    <adapter>/final_adapter/    last finished adapter (PEFT format)
    <adapter>/progress.json     task-level ledger (which problems were already trained on)
    tuning/best_params.json     best Optuna parameters, when a study ran
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .checkpoints import find_latest_checkpoint, list_checkpoints, read_best_adapter
from .fsio import atomic_write_json, publish_dir, read_json, safe_rmtree
from .ledger import LEDGER_FILE
from .secrets_scan import format_findings, scan_dir

logger = logging.getLogger(__name__)

BUNDLE_FILE = "bundle.json"
BUNDLE_SCHEMA = 1


# --------------------------------------------------------------------------------------------- discovery
def discover_bundles(roots: Iterable[str | Path] = ("/kaggle/input",), max_depth: int = 5) -> list[Path]:
    """Attached datasets that contain a `bundle.json`, newest session first."""
    found: list[tuple[int, float, Path]] = []
    for root in roots:
        r = Path(root)
        if not r.is_dir():
            continue
        base_depth = len(r.parts)
        for dirpath, dirnames, filenames in os.walk(r):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            if BUNDLE_FILE in filenames:
                meta = read_json(Path(dirpath) / BUNDLE_FILE, {})
                if isinstance(meta, dict) and meta.get("schema"):
                    found.append((int(meta.get("sessions_total", 0)), float(meta.get("created_ts", 0.0)), Path(dirpath)))
                dirnames[:] = []  # a bundle is a leaf for discovery purposes
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in found]


def resume_candidates(
    run_checkpoints_dir: str | Path,
    adapter: str,
    *,
    resume_from: str | None = None,
    bundle_dirs: Iterable[str | Path] = (),
) -> list[Path]:
    """Resumable `step_<n>/` checkpoints, best first: this run's (crash restart), then `resume_from`, then bundles."""
    found: list[Path] = []

    def add(root: Path) -> None:
        for cand in reversed(list_checkpoints(root)):
            if cand not in found:
                found.append(cand)

    add(Path(run_checkpoints_dir) / adapter)
    if resume_from:
        rp = Path(resume_from)
        if (rp / "trainer_resume.json").is_file() and rp not in found:
            found.append(rp)
        for root in (rp / "checkpoints" / adapter, rp / adapter, rp):
            add(root)
    for b in bundle_dirs:
        add(Path(b) / adapter)
    return found


def best_params_from_bundles(bundle_dirs: Iterable[str | Path]) -> dict[str, Any] | None:
    for b in bundle_dirs:
        data = read_json(Path(b) / "tuning" / "best_params.json")
        if isinstance(data, dict) and isinstance(data.get("params"), dict):
            return data
    return None


# --------------------------------------------------------------------------------------------- writing
def _copy_tree(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".tmp_{dst.name}")
    safe_rmtree(tmp)
    shutil.copytree(src, tmp)
    publish_dir(tmp, dst)


def sync_bundle(
    bundle_dir: str | Path,
    *,
    run_checkpoints_dir: str | Path,
    adapters: Iterable[str],
    served_model_id: str,
    dataset_id: str | None = None,
    dataset_title: str = "GemmaFineTuning LoRA checkpoints",
    run_id: str | None = None,
    git_sha: str | None = None,
    extra: dict[str, Any] | None = None,
    tuning_dir: str | Path | None = None,
    previous_sessions: int = 0,
    scan_secrets: bool = True,
) -> dict[str, Any]:
    """Copies the newest checkpoint / best / final adapter / ledger of every adapter into `bundle_dir`.

    Never raises for a single adapter's problem: each adapter reports its own `status`, so one bad copy cannot
    lose the others. Returns the `bundle.json` content.
    """
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name in adapters:
        src = Path(run_checkpoints_dir) / name
        entry: dict[str, Any] = {"status": "empty"}
        try:
            latest = find_latest_checkpoint(src)
            if latest is not None:
                for old in list_checkpoints(root / name):  # keep exactly one resumable checkpoint per adapter
                    if old.name != latest.name:
                        safe_rmtree(old)
                _copy_tree(latest, root / name / latest.name)
                entry.update(status="ok", checkpoint=latest.name)
            best = read_best_adapter(src)
            if best is not None:
                _copy_tree(best[0], root / name / "best_adapter")
                entry["best"] = best[1]
            if (src / "final_adapter" / "adapter_model.safetensors").is_file():
                _copy_tree(src / "final_adapter", root / name / "final_adapter")
                entry["final_adapter"] = True
                entry.setdefault("status", "ok")
                entry["status"] = "ok"
            if (src / LEDGER_FILE).is_file():
                shutil.copy2(src / LEDGER_FILE, root / name / LEDGER_FILE)
            elif latest is not None and (latest / LEDGER_FILE).is_file():
                shutil.copy2(latest / LEDGER_FILE, root / name / LEDGER_FILE)
        except Exception as exc:  # noqa: BLE001
            entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            logger.error("Bundle sync failed for adapter %r: %s", name, exc)
        summary[name] = entry

    if tuning_dir and Path(tuning_dir).is_dir():
        try:
            copy_tuning_files(Path(tuning_dir), root / "tuning")
        except Exception as exc:  # noqa: BLE001
            logger.error("Bundle sync failed for tuning dir: %s", exc)

    if dataset_id:
        atomic_write_json(
            root / "dataset-metadata.json",
            {"title": dataset_title, "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]},
        )

    findings = scan_dir(root) if scan_secrets else []
    if findings:
        logger.error("SECRETS FOUND in the checkpoint bundle; the files are removed from the bundle:\n%s", format_findings(findings))
        for f in findings:
            target = root / f.path
            if target.is_file():
                target.unlink(missing_ok=True)

    meta = {
        "schema": BUNDLE_SCHEMA,
        "created_ts": time.time(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sessions_total": int(previous_sessions) + 1,
        "run_id": run_id,
        "git_sha": git_sha,
        "served_model_id": served_model_id,
        "adapters": summary,
        "secrets_removed": [f.path for f in findings],
        **(extra or {}),
    }
    atomic_write_json(root / BUNDLE_FILE, meta)
    return meta


TUNING_FILES = ("study.db", "best_params.json", "trials.json")


def copy_tuning_files(src: Path, dst: Path) -> list[str]:
    """Study database and results only: trial run directories hold checkpoints and caches that must not enter the bundle.

    Copying a bundle's `tuning/` onto itself is a no-op. Returns the copied file names."""
    copied: list[str] = []
    dst.mkdir(parents=True, exist_ok=True)
    for name in TUNING_FILES:
        f = src / name
        if f.is_file() and f.resolve() != (dst / name).resolve():
            shutil.copy2(f, dst / name)
            copied.append(name)
    return copied


def tuning_source(explicit_dir: str | Path | None, bundle_dirs: Iterable[str | Path]) -> Path | None:
    """Where the next bundle takes its `tuning/` from: this run's study dir, else the newest attached bundle that has one."""
    if explicit_dir and Path(explicit_dir).is_dir():
        return Path(explicit_dir)
    for b in bundle_dirs:
        t = Path(b) / "tuning"
        if any((t / n).is_file() for n in TUNING_FILES):
            return t
    return None


def previous_session_count(bundle_dirs: Iterable[str | Path]) -> int:
    for b in bundle_dirs:
        meta = read_json(Path(b) / BUNDLE_FILE, {})
        if isinstance(meta, dict) and meta.get("schema"):
            return int(meta.get("sessions_total", 0))
    return 0
