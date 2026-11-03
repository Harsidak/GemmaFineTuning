"""REPORT.md sections that turn a run's artifacts into answers: learning curves, smoke-eval metrics, and a diagnosis
with concrete next steps. Pure functions over run files; every section degrades to a short note instead of raising."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..runctx import RunContext


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                out.append(json.loads(line))
    return out


def train_rows(ctx: RunContext, adapter: str) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(ctx.metrics_dir / f"{adapter}.jsonl") if isinstance(r.get("train_loss"), (int, float))]


def val_rows(ctx: RunContext, adapter: str) -> list[dict[str, Any]]:
    """Validation points from `metrics/<adapter>_val.jsonl` (both backends write it), sorted by step."""
    rows = [r for r in read_jsonl(ctx.metrics_dir / f"{adapter}_val.jsonl") if isinstance(r.get("eval_loss"), (int, float))]
    return sorted(rows, key=lambda r: int(r.get("step") or 0))


# ----------------------------------------------------------------------------------------------- learning curves
def plot_curves(ctx: RunContext, adapter: str) -> Path | None:
    """`plots/<adapter>_curves.png`: train + val loss, grad norm, lr, tokens/s over optimizer steps (None without
    matplotlib or without logged steps)."""
    rows = train_rows(ctx, adapter)
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return None
    vals = val_rows(ctx, adapter)
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].plot(steps, [r["train_loss"] for r in rows], label="train")
    if vals:
        ax[0][0].plot([v["step"] for v in vals], [v["eval_loss"] for v in vals], "o-", label="val")
    ax[0][0].set(title="loss", xlabel="step")
    ax[0][0].legend()
    ax[0][1].plot(steps, [r.get("grad_norm") or 0.0 for r in rows])
    ax[0][1].set(title="grad norm", xlabel="step")
    ax[1][0].plot(steps, [r.get("lr") or 0.0 for r in rows])
    ax[1][0].set(title="learning rate", xlabel="step")
    ax[1][1].plot(steps, [r.get("tokens_per_sec") or 0.0 for r in rows])
    ax[1][1].set(title="tokens / s", xlabel="step")
    fig.suptitle(f"{ctx.run_id} - {adapter}")
    fig.tight_layout()
    out = ctx.run_dir / "plots" / f"{adapter}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def learning_curve_lines(ctx: RunContext, cfg: PipelineConfig) -> list[str]:
    lines = ["## 4. Learning Curves", ""]
    for spec in cfg.adapters:
        rows, vals = train_rows(ctx, spec.name), val_rows(ctx, spec.name)
        if not rows and not vals:
            continue
        lines.append(f"**`{spec.name}`**: {len(rows)} train logs, {len(vals)} validation points")
        if vals:
            best = min(vals, key=lambda v: v["eval_loss"])
            lines.extend(["", "| Step | Val loss | Train loss (nearest log) |", "| ---: | ---: | ---: |"])
            for v in vals:
                near = min(rows, key=lambda r: abs(int(r["step"]) - int(v["step"])))["train_loss"] if rows else None
                mark = " **(best)**" if v is best else ""
                near_s = f"{near:.4f}" if isinstance(near, (int, float)) else "—"
                lines.append(f"| {v['step']} | {v['eval_loss']:.4f}{mark} | {near_s} |")
        try:
            png = plot_curves(ctx, spec.name)
        except Exception:  # noqa: BLE001
            png = None
        if png is not None:
            lines.extend(["", f"![{spec.name} curves]({png.relative_to(ctx.run_dir).as_posix()})"])
        lines.append("")
    if len(lines) == 2:
        lines.extend(["_No training steps were logged._", ""])
    return lines


# ----------------------------------------------------------------------------------------------- smoke eval
def eval_metrics(score: dict[str, Any]) -> dict[str, Any]:
    rows = score.get("tasks") or []
    n = len(rows)
    if not n:
        return {"tasks": 0}
    turns = [r["turns"] for r in rows if isinstance(r.get("turns"), int)]
    errs = [r["tool_call_errors"] for r in rows if isinstance(r.get("tool_call_errors"), int)]
    return {
        "tasks": n,
        "passed": sum(1 for r in rows if r.get("result") == "PASS"),
        "timeouts": sum(1 for r in rows if r.get("result") == "TIMEOUT"),
        "patch_applied": sum(1 for r in rows if r.get("patch_applied") is True),
        "crashed": sum(1 for r in rows if r.get("error") and r.get("turns") is None),
        "mean_turns": round(sum(turns) / len(turns), 1) if turns else None,
        "tool_call_errors": sum(errs) if errs else None,
    }


def smoke_eval_lines(ctx: RunContext, cfg: PipelineConfig) -> list[str]:
    lines = ["## 5. Smoke Evaluation", ""]
    if not ctx.eval_score_path.is_file():
        st = ctx.read_status("eval") or {}
        why = "; ".join(st.get("notes") or []) or st.get("error") or "eval did not run"
        return [*lines, f"_Smoke evaluation did not run: {why}_", ""]
    score = json.loads(ctx.eval_score_path.read_text(encoding="utf-8"))
    m = eval_metrics(score)
    lines.append(
        f"> {m['tasks']} competition task(s) from `{score.get('tasks_file') or cfg.data.tasks_file}`. This checks that "
        "the adapter loads, the agent runs end-to-end and its patches apply; a handful of tasks cannot rank models."
    )
    lines.append("")
    if m["tasks"]:
        lines.append(
            f"**Pass@1 {m['passed']}/{m['tasks']}** | patch applied {m['patch_applied']}/{m['tasks']} | "
            f"timeouts {m['timeouts']} | crashed {m['crashed']} | mean turns {m['mean_turns'] if m['mean_turns'] is not None else '—'} | "
            f"tool-call errors {m['tool_call_errors'] if m['tool_call_errors'] is not None else '—'} | "
            f"vLLM ready in {score.get('vllm_ready_sec', '—')} s"
        )
        lines.append("")
    if score.get("banner"):
        lines.extend(["```text", score["banner"], "```", ""])
    return lines


# ----------------------------------------------------------------------------------------------- diagnosis
def diagnosis(ctx: RunContext, cfg: PipelineConfig, manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(severity, finding, next step) rows. Severity: FAIL > WARN > INFO."""
    out: list[tuple[str, str, str]] = []
    st = {s.get("phase"): s for s in ctx.read_all_statuses()}

    for phase in ("setup", "preflight"):
        s = st.get(phase) or {}
        if s.get("state") in {"failed", "timeout"}:
            hint = ("Check the record format (README: Training data format) and run "
                    "`uv run python scripts/validate_dataset.py <files>` locally before launching."
                    if phase == "preflight" else "Read logs/setup.log: missing model / wheelhouse / GPU.")
            out.append(("FAIL", f"`{phase}` {s.get('state')}: {str(s.get('error') or '')[:300]}", hint))
    for spec in cfg.adapters:
        s = st.get(f"train:{spec.name}") or {}
        note = "; ".join(s.get("notes") or [])
        if s.get("state") == "skipped" and "preflight" in note:
            out.append(("WARN", f"`{spec.name}` was not trained ({note})",
                        f"Give it matching records (roles {spec.roles}) or remove it from `adapters`."))
        elif s.get("state") in {"failed", "timeout"}:
            out.append(("FAIL", f"`train:{spec.name}` {s.get('state')}: {str(s.get('error') or '')[:300]}",
                        f"Read logs/train_{spec.name}.log (OOM -> lower train.max_seq_length; NaN -> lower train.lr)."))

        rows, vals = train_rows(ctx, spec.name), val_rows(ctx, spec.name)
        if rows and len(vals) <= 1:
            out.append(("WARN", f"`{spec.name}`: {len(vals)} validation point(s), so there is no val curve",
                        "Set train.val_every_steps (e.g. steps / 6) so over/under-fitting is visible."))
        if rows and vals:
            best = min(vals, key=lambda v: v["eval_loss"])
            last_train = sum(r["train_loss"] for r in rows[-3:]) / len(rows[-3:])
            if last_train < 0.6 * best["eval_loss"]:
                gap = f"`{spec.name}`: train loss {last_train:.3f} vs best val {best['eval_loss']:.3f} (gap > 40%): likely over-fitting"
                out.append(("WARN", gap, "Fewer epochs, more / more varied data, or lora.dropout 0.05."))
            if vals[-1]["eval_loss"] > best["eval_loss"] * 1.02 and best is not vals[-1]:
                out.append(("INFO", f"`{spec.name}`: val loss bottomed at step {best['step']} and rose after it",
                            "train.select_best keeps that checkpoint; next run can stop near that step."))
            elif best is vals[-1] and len(vals) >= 3:
                out.append(("INFO", f"`{spec.name}`: val loss was still falling at the last point",
                            "More steps/epochs or data are likely to help if the time budget allows."))
        gns = [float(r["grad_norm"]) for r in rows if isinstance(r.get("grad_norm"), (int, float))]
        if gns and max(gns) > 10:
            out.append(("WARN", f"`{spec.name}`: grad-norm spike {max(gns):.1f}", "Lower train.lr or check for very long examples."))

        planned = s.get("planned_duration_sec") or 0
        actual = s.get("actual_duration_sec") or 0
        if s.get("state") == "ok" and planned > 3600 and actual < 0.5 * planned:
            out.append(("INFO", f"`{spec.name}` used {actual / 3600:.1f} h of a {planned / 3600:.1f} h training budget",
                        "Room for more data or epochs (raise train.max_epochs only if the val curve is still falling)."))

    ev = st.get("eval") or {}
    if ev.get("state") == "skipped":
        out.append(("WARN", "Eval skipped: " + "; ".join(ev.get("notes") or []), "See the note; the zip is unaffected."))
    elif ev.get("state") in {"failed", "timeout"}:
        out.append(("WARN", f"Eval {ev.get('state')}: {str(ev.get('error') or '')[:300]}", "Read logs/eval.log; the zip is unaffected."))
    for n in ev.get("notes") or []:
        if "SMOKE TASKS NOT RUN" in n:
            out.append(("WARN", n, "Attach the competition dataset so tasks.jsonl and snapshots/ are mounted."))
    if ctx.eval_score_path.is_file():
        m = eval_metrics(json.loads(ctx.eval_score_path.read_text(encoding="utf-8")))
        if m["tasks"]:
            if m["patch_applied"] < m["tasks"] / 2:
                out.append(("WARN", f"Patches applied on only {m['patch_applied']}/{m['tasks']} tasks",
                            "Inspect eval/results/*: the agent may not call submit_patch, or edits break the diff."))
            if m["timeouts"]:
                out.append(("WARN", f"{m['timeouts']} task(s) timed out", "Raise eval budget / cut turns (prompts, thinking_budget)."))
            if m["tool_call_errors"]:
                out.append(("WARN", f"{m['tool_call_errors']} tool-call error(s)", "Check tool-call format in training data and prompts."))
            if m["patch_applied"] and not m["passed"]:
                out.append(("INFO", "Patches apply but no hidden test passed", "Model quality, not plumbing: more / better data."))
    for aname, pres in (manifest.get("parity") or {}).items():
        if pres.get("verdict") == "SUSPECT":
            out.append(("FAIL", f"Parity SUSPECT on `{aname}`", str(pres.get("recommendation") or "")))

    from ..shared.sysmon import summarize as summarize_system

    sysum = summarize_system(ctx.metrics_dir / "system.jsonl") or {}
    disk_min = sysum.get("disk_free_gib_min")
    if isinstance(disk_min, (int, float)) and disk_min < 10:
        out.append(("WARN", f"Free disk fell to {disk_min} GiB", "Lower train.keep_checkpoints or session data size."))
    summaries = manifest.get("training_summaries") or {}
    track = {(((summaries.get(a.name) or {}).get("session") or {}).get("tracking") or {}).get("mode") for a in cfg.adapters}
    if "offline" in track:
        out.append(("INFO", "W&B logged offline", "`uv run scripts/kaggle_push.py -<pipeline> -output` syncs it (or `wandb sync`)."))
    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    return sorted(out, key=lambda r: order[r[0]])


def diagnosis_lines(ctx: RunContext, cfg: PipelineConfig, manifest: dict[str, Any]) -> list[str]:
    rows = diagnosis(ctx, cfg, manifest)
    lines = ["## Diagnosis & Next Steps", ""]
    if not rows:
        return [*lines, "_Nothing flagged._", ""]
    lines.extend(["| | Finding | Next step |", "| :--- | :--- | :--- |"])
    for sev, finding, step in rows:
        lines.append(f"| **{sev}** | {finding.replace('|', '/')} | {step.replace('|', '/')} |")
    lines.append("")
    return lines
