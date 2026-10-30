"""`TrainingSession`: the one object both trainer backends use for everything that is not the math.

TRL (PyTorch) and Tunix (JAX) differ only in how they run a step and serialise tensors. Resume discovery and
compatibility, the task ledger, NaN / divergence guards, best-checkpoint selection, tracking, the adapter contract
check and the end-of-run summary all live here, so there is exactly one implementation of each.

    session = TrainingSession(...)
    info    = session.resolve_resume()          # newest compatible checkpoint (or None)
    plan    = session.plan_data(info)           # which problems to skip because they were already trained on
    ...
    session.check_health(step, loss, grad_norm) # guards (every rank)
    session.log_metrics(step, {...})            # tracking (rank 0)
    session.on_eval(step, val_loss, save_best)  # best-adapter bookkeeping, early stop
    session.checkpoint_meta() / session.write_progress(tmp, epoch, consumed_keys)   # inside write_checkpoint()
    session.finalize(final_adapter_dir)
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import discover_bundles, resume_candidates
from .checkpoints import ADAPTER_FILE, RESUME_FILE, checkpoint_is_loadable, read_best_adapter, write_best_adapter
from .compat import (
    Problem,
    ResumeIncompatible,
    ResumeVerdict,
    compare_data,
    compare_fingerprints,
    data_fingerprint,
    training_fingerprint,
)
from .fsio import atomic_write_json, read_json
from .guards import BestDecision, BestTracker, LossSpikeGuard, NonFiniteGuard
from .ledger import ProgressLedger, ResumePlan, example_keys
from .tracking import Tracker

logger = logging.getLogger(__name__)


@dataclass
class ResumeInfo:
    checkpoint: Path
    meta: dict[str, Any]
    verdict: ResumeVerdict
    explicit: bool
    problems: list[Problem] = field(default_factory=list)

    @property
    def step(self) -> int:
        return int(self.meta.get("step", 0))

    @property
    def optimizer_compatible(self) -> bool:
        return self.verdict.optimizer_compatible


class TrainingSession:
    def __init__(
        self,
        *,
        run_checkpoints_dir: str | Path,
        run_dir: str | Path,
        cfg: Any,
        eff: Any,
        adapter_name: str,
        backend: str,
        accelerator: str,
        is_main: bool,
        train_rows: Sequence[dict[str, Any]],
        tokenizer_sha: str | None = None,
        template_sha: str | None = None,
        train_base_name: str | None = None,
    ) -> None:
        self.cfg, self.eff = cfg, eff
        self.adapter = adapter_name
        self.backend, self.accelerator = backend, accelerator
        self.is_main = is_main
        self.run_dir = Path(run_dir)
        self.ckpt_root = Path(run_checkpoints_dir) / adapter_name
        self.label = f"[{adapter_name}]"

        self.keys: list[str] = example_keys(train_rows)
        self.data_fp = data_fingerprint(
            self.keys, tokenizer_sha=tokenizer_sha, template_sha=template_sha, max_seq_length=eff.train.max_seq_length
        )
        self.fingerprint = training_fingerprint(
            adapter=adapter_name,
            served_model_id=cfg.effective_served_model_id(),
            train_base=train_base_name,
            quantization=eff.model_quantization,
            backend=backend,
            lora={
                "r": eff.lora.r, "alpha": eff.lora.alpha, "target_regex": eff.lora.target_regex,
                "use_rslora": eff.lora.use_rslora, "exclude_kv_shared": eff.lora.exclude_kv_shared,
            },
        )
        t = eff.train
        self.nonfinite = NonFiniteGuard(t.max_nonfinite_logs, label=self.label)
        self.spike = LossSpikeGuard(t.loss_spike_factor, label=self.label)
        self.best = BestTracker(t.early_stop_patience, t.early_stop_min_delta)
        self.ledger = ProgressLedger(adapter_name)
        self.tracker: Tracker | None = None
        self.resume: ResumeInfo | None = None
        self.plan: ResumePlan | None = None
        self.notes: list[str] = []
        self.early_stop_requested = False

    # ------------------------------------------------------------------ resume
    def bundle_dirs(self) -> list[Path]:
        explicit = [Path(p) for p in self.cfg.resume.bundle_dirs]
        found = discover_bundles([self.cfg.resume.input_root]) if self.cfg.resume.auto_discover else []
        return [*explicit, *[p for p in found if p not in explicit]]

    def resolve_resume(self) -> ResumeInfo | None:
        """Newest loadable, compatible checkpoint. An explicitly requested incompatible one raises; an auto-found one is skipped."""
        explicit_roots = [Path(p).resolve() for p in [self.cfg.run.resume_from, *self.cfg.resume.bundle_dirs] if p]
        cands = resume_candidates(
            self.ckpt_root.parent, self.adapter, resume_from=self.cfg.run.resume_from, bundle_dirs=self.bundle_dirs()
        )
        skipped: list[str] = []
        for cand in cands:
            if not checkpoint_is_loadable(cand):
                continue
            meta = read_json(cand / RESUME_FILE, {}) or {}
            explicit = any(str(cand.resolve()).startswith(str(root)) for root in explicit_roots)
            verdict = compare_fingerprints(meta.get("fingerprint"), self.fingerprint)
            if verdict.fatal:
                msg = f"{cand}: " + "; ".join(p.message for p in verdict.fatal)
                if explicit and self.cfg.resume.strict:
                    raise ResumeIncompatible(
                        "Refusing to resume " + msg + ". Clear run.resume_from / the attached bundle, or restore the old settings."
                    )
                skipped.append(msg)
                logger.warning("%s skipping incompatible checkpoint %s", self.label, msg)
                continue
            problems = [*verdict.warnings, *compare_data(meta.get("data_fingerprint"), self.data_fp)]
            for p in problems:
                logger.warning("%s resume note (%s): %s", self.label, p.code, p.message)
            self.resume = ResumeInfo(cand, meta, verdict, explicit, problems)
            self.notes += [f"resume {p.code}: {p.message}" for p in problems]
            if meta.get("best"):
                self.best.load_state_dict(meta["best"])
            if meta.get("spike"):
                self.spike.load_state_dict(meta["spike"])
            return self.resume
        if skipped:
            self.notes.append("skipped incompatible checkpoint(s): " + " | ".join(skipped))
        return None

    def plan_data(self, info: ResumeInfo | None) -> ResumePlan | None:
        """Ledger-based plan, or None when the checkpoint predates the ledger (callers fall back to the sampler offset)."""
        if info is None:
            return None
        led = ProgressLedger.load(self.adapter, info.checkpoint)
        if led is None:
            self.notes.append("checkpoint has no progress ledger; resuming by sampler offset")
            return None
        self.ledger = led
        self.plan = led.plan(self.keys, on_data_change=self.cfg.resume.on_data_change)
        logger.info("%s data plan: %s", self.label, self.plan.note)
        self.notes.append(f"data plan: {self.plan.note}")
        if self.plan.stale_keys:
            self.notes.append(f"{self.plan.stale_keys} previously trained example(s) are no longer in the dataset")
        return self.plan

    # ------------------------------------------------------------------ tracking
    def start_tracking(self, config_dict: dict[str, Any] | None = None) -> None:
        tr = self.cfg.tracking
        self.tracker = Tracker(
            requested=tr.wandb,
            tuning_active=bool(self.cfg.tuning.active),
            is_main=self.is_main,
            run_dir=self.run_dir,
            project=tr.project,
            run_name=tr.run_name or f"{self.run_dir.name}-{self.adapter}",
            group=tr.group,
            config=config_dict or {"fingerprint": self.fingerprint},
            disable_during_tuning=tr.disable_during_tuning,
        )

    # ------------------------------------------------------------------ per-step hooks (called by TRL callback / Tunix loop)
    def check_health(self, step: int, loss: float | None, grad_norm: float | None = None) -> None:
        """NaN/Inf and divergence guards; call on EVERY rank so all ranks abort together. Raises NonFiniteLossError /
        LossDivergedError, which make the orchestrator restart from the last good checkpoint."""
        self.nonfinite.observe(loss, grad_norm, step)
        if loss is not None:
            self.spike.observe(float(loss), step)

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if self.tracker is not None:
            self.tracker.log(step, metrics)

    def on_eval(self, step: int, val_loss: float, save_best: Callable[[Path], None] | None = None) -> BestDecision:
        decision = self.best.update(step, val_loss)
        if decision.is_best and save_best is not None and self.is_main:
            try:
                write_best_adapter(self.ckpt_root, step, val_loss, save_best)
            except Exception as exc:  # noqa: BLE001 - a failed best-copy must not stop training
                logger.warning("%s could not save best adapter at step %d: %s", self.label, step, exc)
        if decision.should_stop:
            self.early_stop_requested = True
            logger.info("%s early stopping: %d validations without improvement (best %.5f @ step %s)",
                        self.label, decision.evals_without_improvement, decision.best_loss or float("nan"), decision.best_step)
        if self.tracker is not None:
            self.tracker.log(step, {"val/loss": val_loss, "val/best": decision.best_loss})
        return decision

    # ------------------------------------------------------------------ checkpoint content shared by both backends
    def checkpoint_meta(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "data_fingerprint": self.data_fp,
            "best": self.best.state_dict(),
            "spike": self.spike.state_dict(),
            "backend": self.backend,
        }

    def write_progress(self, directory: str | Path, epoch: int, consumed_keys: Sequence[str]) -> None:
        self.ledger.snapshot(epoch=epoch, consumed_keys=consumed_keys, data_sha=self.data_fp["keys_sha"], data_count=self.data_fp["count"])
        self.ledger.save(directory)

    # ------------------------------------------------------------------ end of run
    def finalize(self, final_adapter_dir: str | Path, *, steps_completed: int, summary: dict[str, Any]) -> dict[str, Any]:
        """Best-vs-last adapter selection, adapter contract check, ledger / tracker close. Never raises."""
        out: dict[str, Any] = {}
        final_dir = Path(final_adapter_dir)
        try:
            best = read_best_adapter(self.ckpt_root)
            if self.eff.train.select_best and best is not None and (final_dir / ADAPTER_FILE).is_file():
                best_dir, best_meta = best
                last_val = summary.get("final_val_loss")
                if last_val is None or float(best_meta["val_loss"]) < float(last_val):
                    shutil.copy2(best_dir / ADAPTER_FILE, final_dir / ADAPTER_FILE)
                    out.update(selected="best", selected_step=best_meta["step"], selected_val_loss=best_meta["val_loss"])
                    logger.info("%s final adapter = best checkpoint (step %s, val %.5f)", self.label, best_meta["step"], best_meta["val_loss"])
                else:
                    out["selected"] = "last"
            elif best is not None:
                out["selected"] = "last"
            out["best"] = {"step": self.best.best_step, "val_loss": self.best.best_loss}
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s best-adapter selection failed: %s", self.label, exc)
            out["selection_error"] = str(exc)
        try:
            from .adapter_contract import check_adapter_contract

            served = self.cfg.model.served_path
            rep = check_adapter_contract(
                final_dir, served_model_dir=served if served and served != "__tiny__" and Path(served).is_dir() else None,
                expected_rank=self.eff.lora.r,
            )
            out["contract"] = rep.to_dict()
            if not rep.ok:
                logger.error("%s ADAPTER CONTRACT FAILED: %s", self.label, "; ".join(rep.errors[:5]))
        except Exception as exc:  # noqa: BLE001
            out["contract"] = {"ok": None, "error": f"{type(exc).__name__}: {exc}"}
        try:
            self.ledger.end_session(steps=steps_completed, backend=self.backend, accelerator=self.accelerator, notes=self.notes[-5:])
            self.ledger.save(self.ckpt_root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s could not write the final ledger: %s", self.label, exc)
        if self.tracker is not None:
            self.tracker.summary({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))})
            self.tracker.finish()
            out["tracking"] = {"mode": self.tracker.decision.mode, "reason": self.tracker.decision.reason, "failed": self.tracker.failed}
        if self.is_main:
            atomic_write_json(self.ckpt_root / "session_summary.json", {**summary, **out, "notes": self.notes})
        return out


def load_session_summary(ckpt_root: str | Path) -> dict[str, Any]:
    return read_json(Path(ckpt_root) / "session_summary.json", {}) or {}

