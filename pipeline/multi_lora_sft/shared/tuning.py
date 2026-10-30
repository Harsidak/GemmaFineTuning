"""Optuna hyper-parameter search that works the same for the TRL (GPU) and Tunix (TPU) backends.

Each trial is a normal, complete run of the orchestrator in a subprocess (`python -m pipeline.multi_lora_sft.run`)
with `tuning.active=true` (W&B off, no export / package / eval / bundle) and a short wall-clock window. The tuner never
imports torch or jax: it chooses parameters, launches the run, watches `metrics/<adapter>.jsonl` to prune hopeless
trials early, and reads the validation loss from the run's manifest. Because the trainer is whatever the config
resolves to, one search covers either backend.

* The study lives in SQLite (`<storage_dir>/study.db`) and resumes across sessions; it is copied into the checkpoint
  bundle (`tuning/`), so the next session continues the same study.
* `best_params.json` is written after every completed trial, so a killed session still leaves usable results.
* Trials share one tokenization cache (the cache key does not depend on lr / rank / schedule).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fsio import atomic_write_json, read_json

logger = logging.getLogger(__name__)

DEFAULT_SPACE: dict[str, dict[str, Any]] = {
    "train.lr": {"type": "float", "low": 3.0e-5, "high": 3.0e-4, "log": True},
    "lora.r": {"type": "categorical", "choices": [16, 32, 64]},
    "lora.alpha_per_r": {"type": "categorical", "choices": [1, 2]},
    "schedule.warmup_steps": {"type": "int", "low": 5, "high": 40},
    "schedule.decay_frac": {"type": "float", "low": 0.05, "high": 0.30},
    "train.weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
}
ALPHA_KEY = "lora.alpha_per_r"


class TuningError(RuntimeError):
    pass


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TuningError("optuna is not installed (attach the banwait13/gemmafinetunewheels bundle or `pip install optuna`)") from exc
    return optuna


def validate_space(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, spec in space.items():
        if not isinstance(spec, dict) or spec.get("type") not in {"float", "int", "categorical"}:
            raise TuningError(f"tuning.space[{key!r}] needs a 'type' of float | int | categorical, got {spec!r}")
        s = dict(spec)
        if s["type"] == "categorical":
            if not s.get("choices"):
                raise TuningError(f"tuning.space[{key!r}] categorical needs non-empty 'choices'")
        else:
            try:
                s["low"], s["high"] = (float(s["low"]), float(s["high"])) if s["type"] == "float" else (int(s["low"]), int(s["high"]))
            except (KeyError, TypeError, ValueError) as exc:  # YAML reads '1e-5' as a string; float() accepts it
                raise TuningError(f"tuning.space[{key!r}] needs numeric 'low' and 'high': {exc}") from exc
            if s["low"] > s["high"]:
                raise TuningError(f"tuning.space[{key!r}] has low > high")
            if s.get("log") and s["low"] <= 0:
                raise TuningError(f"tuning.space[{key!r}] log scale needs low > 0")
        out[key] = s
    return out


def suggest_params(trial: Any, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, s in space.items():
        if s["type"] == "categorical":
            params[key] = trial.suggest_categorical(key, list(s["choices"]))
        elif s["type"] == "int":
            params[key] = trial.suggest_int(key, s["low"], s["high"], step=int(s.get("step", 1)), log=bool(s.get("log", False)))
        else:
            kw: dict[str, Any] = {"log": bool(s.get("log", False))}
            if s.get("step") is not None and not kw["log"]:
                kw["step"] = float(s["step"])
            params[key] = trial.suggest_float(key, s["low"], s["high"], **kw)
    return params


def params_to_sets(params: dict[str, Any], base_r: int) -> list[str]:
    """`--set key=value` strings for a parameter dict; `lora.alpha_per_r` becomes `lora.alpha = r * multiplier`."""
    p = dict(params)
    sets: list[str] = []
    mult = p.pop(ALPHA_KEY, None)
    for key, val in p.items():
        sets.append(f"{key}={json.dumps(val)}")
    if mult is not None:
        r = int(p.get("lora.r", base_r))
        sets.append(f"lora.alpha={json.dumps(int(round(r * float(mult))))}")
    return sets


def strip_tuned_overrides(overrides: dict[str, Any] | None, tuned_keys: list[str]) -> dict[str, Any]:
    """A per-adapter override would silently beat a tuned global value; drop the ones being tuned."""
    ov = copy.deepcopy(overrides or {})
    for key in [*tuned_keys, *(["lora.alpha"] if ALPHA_KEY in tuned_keys else [])]:
        section, _, name = key.partition(".")
        if isinstance(ov.get(section), dict):
            ov[section].pop(name, None)
            if not ov[section]:
                ov.pop(section)
    return ov


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a half-written last line
    except OSError:
        pass
    return rows


def trial_objective(run_dir: Path, adapter: str) -> tuple[float | None, str]:
    """(value, source): final validation loss, else the mean of the last training losses."""
    manifest = read_json(run_dir / "manifest.json", {}) or {}
    summ = (manifest.get("training_summaries") or {}).get(adapter) or {}
    val = summ.get("final_val_loss")
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val), "val_loss"
    rows = [r for r in read_metrics(run_dir / "metrics" / f"{adapter}.jsonl") if isinstance(r.get("train_loss"), (int, float))]
    tail = [float(r["train_loss"]) for r in rows[-5:] if math.isfinite(float(r["train_loss"]))]
    if tail:
        return sum(tail) / len(tail), "train_loss_tail"
    return None, "none"


@dataclass
class StudyResult:
    best_params: dict[str, Any] | None
    best_value: float | None
    best_sets: list[str]
    completed: int
    pruned: int
    failed: int
    study_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params, "best_value": self.best_value, "best_sets": self.best_sets,
            "completed": self.completed, "pruned": self.pruned, "failed": self.failed, "study_dir": str(self.study_dir),
        }


def build_trial_sets(cfg: Any, adapter: str, tuned_sets: list[str], *, study_dir: Path, trial_no: int, shared_cache: Path | None) -> list[str]:
    """Overrides that turn the normal config into one short, quiet trial run of a single adapter."""
    tu = cfg.tuning
    spec = next(a for a in cfg.adapters if a.name == adapter)
    tuned_keys = [s.split("=", 1)[0] for s in tuned_sets]
    adapter_spec = {"name": spec.name, "roles": spec.roles, "time_share": None, "overrides": strip_tuned_overrides(spec.overrides, tuned_keys)}
    sets = [
        f"run.output_root={json.dumps(str(study_dir / 'trials'))}",
        f"run.run_id={json.dumps(f'trial_{trial_no:04d}')}",
        "run.mode=\"calibration\"",
        f"run.calibration_minutes={json.dumps(float(tu.trial_minutes))}",
        "run.resume_from=null",
        "resume.auto_discover=false",
        "resume.bundle_dirs=[]",
        "tuning.active=true",
        "tuning.enabled=false",
        "eval.enabled=false",
        "output.bundle_dir=null",
        "tracking.wandb=\"off\"",
        f"adapters={json.dumps([adapter_spec])}",
        f"train.val_every_min={json.dumps(max(0.5, round(float(tu.trial_minutes) / 4.0, 2)))}",
        "train.checkpoint_every_min=100000.0",
        "train.max_restarts=0",
        "submission.working_zip_path=null",
        *tuned_sets,
    ]
    if tu.trial_max_steps is not None:
        sets.append(f"train.max_steps={int(tu.trial_max_steps)}")
    if shared_cache is not None and shared_cache.is_dir():
        sets.append(f"data.cache_dir={json.dumps(str(shared_cache))}")
    return sets


def _kill_tree(proc: subprocess.Popen[Any]) -> None:
    from ..run import _terminate_process_tree

    _terminate_process_tree(proc, grace_sec=10.0)


def run_study(
    config_path: str | Path | None,
    base_sets: list[str],
    *,
    python: str | None = None,
    poll_sec: float = 5.0,
    runner: Any = None,
) -> StudyResult:
    """Runs (or resumes) the Optuna study described by `tuning.*` and returns the best parameters.

    `runner(cmd, trial_dir, trial, adapter)` is injectable for tests; the default launches the orchestrator.
    """
    from ..config import load_config

    optuna = _import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(config_path, base_sets)
    tu = cfg.tuning
    adapter = tu.adapter or cfg.adapters[0].name
    space = validate_space(tu.space or DEFAULT_SPACE)
    base_r = cfg.resolve_adapter(adapter).lora.r

    study_dir = Path(tu.storage_dir or (Path(cfg.run.output_root) / "tuning"))
    study_dir.mkdir(parents=True, exist_ok=True)
    for bundle in [Path(p) for p in cfg.resume.bundle_dirs] + _bundle_candidates(cfg):
        prev = bundle / "tuning" / "study.db"
        if prev.is_file() and not (study_dir / "study.db").exists():
            shutil.copy2(prev, study_dir / "study.db")
            logger.info("Resuming Optuna study from %s", prev)
            break

    sampler = optuna.samplers.TPESampler(seed=tu.seed) if tu.sampler == "tpe" else optuna.samplers.RandomSampler(seed=tu.seed)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=tu.n_startup_trials, n_warmup_steps=10)
        if tu.pruner == "median" else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=f"{cfg.run.name}-{adapter}", storage=f"sqlite:///{(study_dir / 'study.db').as_posix()}",
        direction="minimize", sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    t0 = time.time()
    session_cap = cfg.budget.session_hours * 3600.0 - cfg.budget.safety_margin_min * 60.0
    cap_sec = min(tu.total_minutes * 60.0 if tu.total_minutes else session_cap, session_cap)
    shared_cache = study_dir / "shared_cache"
    py = python or sys.executable
    launch = runner or _default_runner

    def objective(trial: Any) -> float:
        params = suggest_params(trial, space)
        sets = build_trial_sets(cfg, adapter, params_to_sets(params, base_r), study_dir=study_dir, trial_no=trial.number,
                                shared_cache=shared_cache)
        cmd = [py, "-m", "pipeline.multi_lora_sft.run"] + (["--config", str(config_path)] if config_path else [])
        for expr in [*base_sets, *sets]:
            cmd += ["--set", expr]
        trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        rc = launch(cmd, trial_dir, trial, adapter, poll_sec)
        value, source = trial_objective(trial_dir, adapter)
        trial.set_user_attr("objective_source", source)
        trial.set_user_attr("exit_code", rc)
        if (trial_dir / "cache").is_dir() and not shared_cache.exists():
            shutil.copytree(trial_dir / "cache", shared_cache, dirs_exist_ok=True)
        if value is None or not math.isfinite(value):
            raise TuningError(f"trial {trial.number} produced no usable loss (exit code {rc}); see {trial_dir}/logs")
        return value

    def after(_study: Any, _trial: Any) -> None:
        _write_results(study, study_dir, adapter, base_r, cfg)

    while len([t for t in study.trials if t.state.is_finished()]) < tu.trials:
        remaining = cap_sec - (time.time() - t0)
        if remaining < tu.trial_minutes * 60.0 + 60.0:
            logger.warning("Stopping the study: %.1f min left, a trial needs %.1f min", remaining / 60.0, tu.trial_minutes)
            break
        study.optimize(objective, n_trials=1, catch=(TuningError, Exception), callbacks=[after])

    return _write_results(study, study_dir, adapter, base_r, cfg)


def _bundle_candidates(cfg: Any) -> list[Path]:
    from .bundle import discover_bundles

    return discover_bundles([cfg.resume.input_root]) if cfg.resume.auto_discover else []


def _write_results(study: Any, study_dir: Path, adapter: str, base_r: int, cfg: Any) -> StudyResult:
    optuna = _import_optuna()
    states = optuna.trial.TrialState
    done = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    failed = [t for t in study.trials if t.state == states.FAIL]
    atomic_write_json(
        study_dir / "trials.json",
        [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, "user_attrs": t.user_attrs} for t in study.trials],
    )
    best_params = best_value = None
    best_sets: list[str] = []
    if done:
        best = min(done, key=lambda t: t.value)
        best_params, best_value = dict(best.params), float(best.value)
        best_sets = params_to_sets(best_params, base_r)
        atomic_write_json(
            study_dir / "best_params.json",
            {"adapter": adapter, "trial": best.number, "value": best_value, "params": best_params, "sets": best_sets,
             "completed": len(done), "pruned": len(pruned), "failed": len(failed), "created_ts": time.time()},
        )
    return StudyResult(best_params, best_value, best_sets, len(done), len(pruned), len(failed), study_dir)


def _default_runner(cmd: list[str], trial_dir: Path, trial: Any, adapter: str, poll_sec: float) -> int:
    """Launches one trial run, feeds intermediate losses to Optuna, and kills the process tree if the trial is pruned."""
    optuna = _import_optuna()
    trial_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir.parent / f"{trial_dir.name}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    popen: dict[str, Any] = {"cwd": root, "env": env, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen["start_new_session"] = True
    last_step = -1
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, **popen)
        while proc.poll() is None:
            time.sleep(poll_sec)
            rows = [r for r in read_metrics(trial_dir / "metrics" / f"{adapter}.jsonl") if isinstance(r.get("train_loss"), (int, float))]
            if rows and int(rows[-1]["step"]) > last_step:
                last_step = int(rows[-1]["step"])
                recent = [float(r["train_loss"]) for r in rows[-3:]]
                if all(math.isfinite(v) for v in recent):
                    trial.report(sum(recent) / len(recent), last_step)
                    if trial.should_prune():
                        _kill_tree(proc)
                        raise optuna.TrialPruned()
        return int(proc.returncode or 0)


def apply_best_params(sets: list[str], bundle_dirs: list[str | Path]) -> tuple[list[str], str | None]:
    """Adds an attached bundle's tuned parameters in front of `sets` (so explicit later overrides still win)."""
    from .bundle import best_params_from_bundles

    best = best_params_from_bundles(bundle_dirs)
    if not best:
        return sets, None
    note = f"tuned params from trial {best.get('trial')} (val {best.get('value')}): {best.get('params')}"
    return [*best.get("sets", []), *sets], note
