"""Phase 7 (`report`): consolidates `manifest.json` and writes human-readable `REPORT.md`.

Implements design_doc.md §6.2, §9.4, and §10.3:
- Always runs last regardless of earlier phase failures or timeouts.
- Summarizes phase statuses/durations, preflight & leakage stats, per-adapter training metrics,
  dead-module checks, parity verdicts (flagging `SUSPECT` prominently), packaged zip sha256/size,
  adapter fallback sources, and the smoke eval table + statistical disclaimer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from ..config import PipelineConfig, load_config
from ..runctx import PhaseStatus, RunContext, configure_phase_logging, utc_iso


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                out.append(json.loads(s))
    return out


def build_report_markdown(
    ctx: RunContext,
    cfg: PipelineConfig,
    manifest: dict[str, Any],
    statuses: list[dict[str, Any]],
) -> str:
    """Renders `REPORT.md` from run artifacts and manifest."""
    lines: list[str] = [
        f"# Multi-LoRA SFT Run Report — `{ctx.run_id}`",
        "",
        f"- **Mode:** `{cfg.run.mode}` | **Seed:** `{cfg.run.seed}` | **Served Model:** `{cfg.effective_served_model_id()}`",
        f"- **Code Hash:** `{manifest.get('code_sha256', 'unknown')[:16]}` | **Git SHA:** `{manifest.get('git_sha') or 'n/a'}`",
        f"- **Session T0:** `{manifest.get('session_t0_iso', 'unknown')}`",
    ]

    zip_path = manifest.get("submission_zip_path")
    zip_sha = manifest.get("submission_zip_sha256")
    zip_bytes = manifest.get("submission_zip_bytes")
    if zip_path and zip_sha:
        mib = float(zip_bytes or 0) / (1024 * 1024)
        lines.append(f"- **Submission Zip:** `{zip_path}` ({mib:.2f} MiB, sha256 `{zip_sha[:16]}...`)")
        if manifest.get("working_zip_path"):
            lines.append(f"- **Working Copy:** `{manifest['working_zip_path']}`")
    else:
        lines.append("- **Submission Zip:** **MISSING (CRITICAL)**")
    lines.append("")

    # Prominent SUSPECT parity alert (§9.3)
    parity_map = manifest.get("parity") or {}
    for aname, pres in parity_map.items():
        if pres.get("verdict") == "SUSPECT":
            lines.extend(
                [
                    (
                        f"> **WARNING — PARITY SUSPECT (`{aname}`):** greedy agreement {pres.get('adapter_greedy_agreement')} "
                        f"vs base {pres.get('base_greedy_agreement')}. Recommendation: "
                        f"**{pres.get('recommendation', 'retrain at r=16 or with fewer target modules (vllm#50059)')}**."
                    ),
                    "",
                ]
            )

    # Phase timeline table
    lines.extend(
        [
            "## 1. Phase Timeline",
            "",
            "| Phase | State | Actual (s) | Planned (s) | Notes / Error |",
            "| :--- | :--- | ---: | ---: | :--- |",
        ]
    )
    for st in statuses:
        ph = st.get("phase", "?")
        state = st.get("state", "?")
        act = f"{st['actual_duration_sec']:.1f}" if st.get("actual_duration_sec") is not None else "—"
        pln = f"{st['planned_duration_sec']:.1f}" if st.get("planned_duration_sec") is not None else "—"
        note_parts = list(st.get("notes") or [])
        if st.get("error"):
            note_parts.insert(0, f"ERROR: {st['error']}")
        note_str = "; ".join(note_parts).replace("|", "\\|")[:160] or "—"
        lines.append(f"| `{ph}` | **{state.upper()}** | {act} | {pln} | {note_str} |")
    lines.append("")

    # Preflight summary
    lines.append("## 2. Preflight & Data Contract")
    lines.append("")
    if ctx.preflight_report_path.is_file():
        pf = json.loads(ctx.preflight_report_path.read_text(encoding="utf-8"))
        leak = pf.get("leakage", {})
        lines.append(
            f"- **Raw trajectories:** {pf.get('raw_trajectory_count', 0)} | "
            f"**Held-out tasks checked:** {pf.get('test_tasks_checked', 0)} | "
            f"**Leakage gate:** `{'PASS' if leak.get('passed') else 'FAIL'}`"
        )
        lines.append(
            f"- **Template sha256:** `{str(pf.get('template_sha256', ''))[:16]}` | "
            f"**Tokenizer sha256:** `{str(pf.get('tokenizer_sha256', ''))[:16]}`"
        )
        lines.extend(
            [
                "",
                "| Adapter | Roles | Train | Val | Sup. Tokens | Sup. % | Overlong | Dup | Length p50 / p90 / max |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for aname, asum in (pf.get("adapters") or {}).items():
            hist = asum.get("length_histogram", {})
            roles_s = ",".join(asum["roles"]) if asum.get("roles") else "all"
            lines.append(
                f"| `{aname}` | `{roles_s}` | {asum.get('train_examples', 0)} | {asum.get('val_examples', 0)} | "
                f"{asum.get('train_supervised_tokens', 0)} | {asum.get('supervised_token_fraction', 0.0) * 100:.1f}% | "
                f"{asum.get('dropped_overlong', 0)} | {asum.get('dropped_duplicates', 0)} | "
                f"{hist.get('p50', 0)} / {hist.get('p90', 0)} / {hist.get('max', 0)} |"
            )
    else:
        lines.append("_Preflight report was not generated._")
    lines.append("")

    # Per-adapter training & packaging summary
    lines.extend(
        [
            "## 3. Adapter Training, Export & Packaging",
            "",
            "| Adapter | Steps | Epochs | Train Loss | Val Loss | Decay? | Dead B | Parity | Packaged Source |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- |",
        ]
    )
    tr_summaries = manifest.get("training_summaries") or {}
    adapter_sources = manifest.get("adapter_sources") or {}
    inv_slot = {v: k for k, v in cfg.submission.adapter_map.items()}

    for spec in cfg.adapters:
        ts = tr_summaries.get(spec.name, {})
        m_rows = _read_jsonl(ctx.metrics_dir / f"{spec.name}.jsonl")
        steps = ts.get("steps_completed", m_rows[-1]["step"] if m_rows else 0)
        epochs = ts.get("epochs_completed", m_rows[-1]["epoch"] if m_rows else 0.0)
        t_loss = ts.get("final_train_loss", m_rows[-1]["train_loss"] if m_rows else None)
        v_loss = ts.get("final_val_loss")
        decay_s = "yes" if ts.get("decay_entered") else "no"
        dead = ts.get("dead_lora_check") or {}
        dead_s = (
            f"{dead.get('dead_lora_b_modules', 0)}/{dead.get('total_lora_b_modules', 0)}"
            if dead
            else "—"
        )
        par_s = (parity_map.get(spec.name) or {}).get("verdict", "—")
        slot_key = inv_slot.get(spec.name, spec.name)
        pkg_src = adapter_sources.get(slot_key, "—")
        t_loss_s = f"{t_loss:.4f}" if isinstance(t_loss, (int, float)) else "—"
        v_loss_s = f"{v_loss:.4f}" if isinstance(v_loss, (int, float)) else "—"
        lines.append(
            f"| `{spec.name}` | {steps} | {epochs} | {t_loss_s} | {v_loss_s} | {decay_s} | {dead_s} | {par_s} | `{pkg_src}` |"
        )
    lines.append("")

    # Smoke eval section + §9.4 1-line statistical meaning note
    lines.append("## 4. Smoke Evaluation")
    lines.append("")
    lines.append(
        "> Smoke eval proves only that the adapter loads, the agent runs end-to-end, and generated patches apply; "
        "a 3-task score cannot rank models—use validation loss, the parity check, and larger Evaluation-track runs."
    )
    lines.append("")
    if ctx.eval_score_path.is_file():
        ev = json.loads(ctx.eval_score_path.read_text(encoding="utf-8"))
        if ev.get("banner"):
            lines.extend(["```text", ev["banner"], "```", ""])
    else:
        lines.append("_Smoke evaluation did not run or was disabled._")
        lines.append("")

    return "\n".join(lines)


def run_report_phase(
    ctx: RunContext,
    cfg: PipelineConfig | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Consolidates all phase statuses and metrics into `manifest.json` and `REPORT.md`."""
    logger = configure_phase_logging(ctx.log_path("report"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)

    statuses = ctx.read_all_statuses()
    manifest = ctx.read_manifest()

    # Build consolidated per-adapter manifest rollups (§10.3)
    per_adapter_manifest: dict[str, Any] = {}
    tr_summaries = manifest.get("training_summaries") or {}
    exported_map = manifest.get("exported_adapters") or {}
    parity_map = manifest.get("parity") or {}
    adapter_sources = manifest.get("adapter_sources") or {}
    inv_slot = {v: k for k, v in cfg.submission.adapter_map.items()}

    for spec in cfg.adapters:
        m_rows = _read_jsonl(ctx.metrics_dir / f"{spec.name}.jsonl")
        tokens_seen = sum(int(r.get("supervised_tokens_in_step", 0)) for r in m_rows)
        slot_key = inv_slot.get(spec.name, spec.name)
        per_adapter_manifest[spec.name] = {
            "training": tr_summaries.get(spec.name),
            "supervised_tokens_logged": tokens_seen,
            "exported": exported_map.get(spec.name),
            "parity": parity_map.get(spec.name),
            "packaged_slot": slot_key,
            "packaged_source": adapter_sources.get(slot_key),
        }

    manifest = ctx.update_manifest(
        {
            "adapters_rollup": per_adapter_manifest,
            "phases": statuses,
            "report_generated_at": utc_iso(),
        }
    )

    report_md = build_report_markdown(ctx, cfg, manifest, statuses)
    ctx.report_path.write_text(report_md, encoding="utf-8")

    t_end = time.time()
    status = PhaseStatus(
        phase="report",
        state="ok",
        started=utc_iso(t_start),
        ended=utc_iso(t_end),
        started_ts=t_start,
        ended_ts=t_end,
        planned_deadline=utc_iso(deadline) if deadline else None,
        planned_deadline_ts=deadline,
        actual_duration_sec=round(t_end - t_start, 2),
        outputs={"report_path": str(ctx.report_path), "manifest_path": str(ctx.manifest_path)},
    )
    ctx.write_status(status)
    # Refresh `phases` in manifest to include `report` itself
    ctx.update_manifest({"phases": ctx.read_all_statuses()})
    logger.info("Wrote REPORT.md (%d chars) and finalized manifest.json", len(report_md))
    return status.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 7: report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    run_report_phase(ctx, deadline=args.deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
