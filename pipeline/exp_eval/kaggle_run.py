"""Kaggle driver: build arms -> start vLLM like the scorer -> run arms x tasks through swegemma -> scorecard.

Run as a subprocess from `kaggle/eval_exp016.ipynb` (a notebook kernel already runs an event loop, so `asyncio.run`
must live in its own process):

    python -m pipeline.exp_eval.kaggle_run --base-model <w4a16 dir> --comp <dir with tasks.jsonl + snapshots/> \
        --out /kaggle/working/exp_eval/<label> --arms ctrl_022,exp016_a0_pruned,exp016_a1_skill_first \
        --tasks dev30 --concurrency 4 --deadline-ts <unix time the notebook must be done by>

The harness calls copy `pipeline/multi_lora_sft/phases/evaluate.py`, whose call shapes ran on Kaggle on 2026-10-31
(that run's traces are what the tally was built against). Every arm is scored with ITS OWN eval_config.yaml budget,
as the scorer does. Arms with identical adapters share one vLLM server; their tasks are interleaved.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .arms import REPO_ROOT, BuiltArm, build_arm, load_yaml
from .runner import run_matrix, select_task_ids
from .tally import render_markdown, scorecard

SERVED_MODEL_ID = "gemma-4-31b-it-qat-w4a16-ct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def adapter_key(sub: Path) -> tuple[tuple[str, str], ...]:
    """Arms whose adapters/ trees are byte-identical can share a vLLM server."""
    root = sub / "adapters"
    if not root.is_dir():
        return ()
    return tuple(sorted((p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in root.rglob("*") if p.is_file()))


def adapter_ranks(sub: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for cfg in sorted((sub / "adapters").glob("*/adapter_config.json")):
        out[cfg.parent.name] = int(json.loads(cfg.read_text(encoding="utf-8")).get("r", 16))
    return out


def served_model_id(sub: Path) -> str:
    agent = load_yaml(sub / "agent.yaml")
    return str(agent.get("model") or SERVED_MODEL_ID)


def build_evaluator(built: BuiltArm, *, comp: Path, registry: Any, manifest: Any, out_dir: Path) -> Any:
    from google.adk.agents.context_cache_config import ContextCacheConfig  # type: ignore[import-not-found]
    from google.adk.apps._configs import EventsCompactionConfig  # type: ignore[import-not-found]
    from swegemma.config import EvalConfig, build_submission_limits  # type: ignore[import-not-found]
    from swegemma.evaluate import Evaluator  # type: ignore[import-not-found]

    budget = built.manifest["eval_config"]  # this arm's eval_config.yaml over the harness defaults
    limits, gen_constraints = build_submission_limits()
    return Evaluator(EvalConfig(
        tasks_path=comp / "tasks.jsonl",
        snapshots_dir=comp / "snapshots",
        results_dir=out_dir / "harness",
        submission_dir=built.submission_dir,
        models=registry,
        sandbox="subprocess",
        timeout_seconds=int(budget["timeout_seconds"]),
        verification_timeout_seconds=300,
        max_time_minutes=float(budget["max_time_minutes"]),
        max_tool_calls=int(budget["max_tool_calls"]),
        max_turns=int(budget["max_turns"]),
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=manifest,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(comp / "graphs") if (comp / "graphs").is_dir() else None,
        embeddings_dir=str(comp / "embeddings") if (comp / "embeddings").is_dir() else None,
        wheels_dir=(comp / "wheels") if (comp / "wheels").is_dir() else None,
        verbose=False,
    ))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--comp", type=Path, required=True, help="competition dir with tasks.jsonl and snapshots/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm ids (experiments/arms/<id>)")
    ap.add_argument("--tasks", default="dev30", help="dev30 | all | smoke<N> | comma-separated task ids")
    ap.add_argument("--control", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--deadline-ts", type=float, default=None, help="unix time by which every task must be done")
    ap.add_argument("--mode", choices=("async", "threads"), default="async")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--arms-root", type=Path, default=REPO_ROOT / "experiments" / "arms")
    args = ap.parse_args(argv)

    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Build and rule-check every arm before anything slow happens.
    built: dict[str, BuiltArm] = {}
    for arm_id in arm_ids:
        built[arm_id] = build_arm(arm_id, args.out / arm_id, arms_root=args.arms_root)
        m = built[arm_id].manifest
        log(f"arm {arm_id}: zip sha256 {m['zip_sha256'][:12]} | budget {m['eval_config']} | "
            f"changed vs {m['base']}: {m['changed_vs_base'] or '-'}")
        for w in m["rule_warnings"]:
            log(f"  WARNING {arm_id}: {w}")
    models = {served_model_id(b.submission_dir) for b in built.values()}
    if len(models) != 1:
        raise SystemExit(f"arms declare different base models {sorted(models)}; run them separately")

    # 2. Tasks.
    from swegemma.models import load_tasks  # type: ignore[import-not-found]

    swe_tasks = {t.instance_id: t for t in load_tasks(args.comp / "tasks.jsonl")}
    repo_of = {tid: str(getattr(t, "repo", "")) for tid, t in swe_tasks.items()}
    dev30 = json.loads((REPO_ROOT / "experiments" / "exp_016_skill_first_agent" / "dev_30_ids.json")
                       .read_text(encoding="utf-8"))["task_ids"]
    named = args.tasks in ("dev30", "all") or args.tasks.startswith("smoke")
    spec: Any = args.tasks if named else [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_ids = select_task_ids(spec, all_ids=sorted(swe_tasks), dev30=dev30, repo_of=repo_of)
    log(f"{len(task_ids)} task(s) x {len(arm_ids)} arm(s), concurrency {args.concurrency}: {', '.join(task_ids)}")
    (args.out / "run_config.json").write_text(json.dumps({**vars(args), "task_ids": task_ids, "arms": arm_ids},
                                                         indent=2, default=str), encoding="utf-8")

    # 3. One vLLM server per distinct adapter set, started the way the scorer starts it.
    import adk_submission  # type: ignore[import-not-found]
    import litellm  # type: ignore[import-not-found]
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS  # type: ignore[import-not-found]

    from ..multi_lora_sft.phases.evaluate import _start_server_failing_fast

    litellm.drop_params = True
    if hasattr(adk_submission, "install_litellm_reasoning_patch"):
        adk_submission.install_litellm_reasoning_patch()
    os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")

    groups: dict[tuple, list[str]] = {}
    for arm_id in arm_ids:
        groups.setdefault(adapter_key(built[arm_id].submission_dir), []).append(arm_id)

    served_id = models.pop()
    for gi, (key, members) in enumerate(groups.items(), 1):
        first = built[members[0]].submission_dir
        manifest = adk_submission.discover_adapters(str(first), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        ranks = adapter_ranks(first)
        vcfg = adk_submission.VllmConfig(
            model=str(args.base_model), host="127.0.0.1", port=8000,
            tool_call_parser="gemma4", reasoning_parser="gemma4",
            max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90,
            enable_auto_tool_choice=True, enable_lora=bool(ranks), max_loras=max(1, len(ranks)),
            max_lora_rank=max(ranks.values()) if ranks else 128, scheduler_reserve_full_isl=False,
            tensor_parallel_size=4, startup_timeout=args.startup_timeout,
            extra_args=["--max-num-seqs", str(args.max_num_seqs)],
        )
        server = adk_submission.VllmServer(vcfg, adapter_manifest=manifest)
        registry = server.create_model_registry(aliases=[served_id], model_prefix="openai/", api_key="EMPTY")
        # Evaluators before the 5+ minute model load, so an API mismatch fails in seconds.
        evaluators = {a: build_evaluator(built[a], comp=args.comp, registry=registry, manifest=manifest,
                                         out_dir=args.out / a) for a in members}
        # A task may only start if it can finish (agent budget + ~5 min verification) before the deadline.
        longest = max(float(built[a].manifest["eval_config"]["max_time_minutes"]) for a in members)
        deadline = None if args.deadline_ts is None else args.deadline_ts - (longest + 6.0) * 60.0
        log(f"server {gi}/{len(groups)}: arms {members}, adapters {sorted(ranks) or 'none'}")
        try:
            t0 = time.time()
            _start_server_failing_fast(server, _Logger())
            log(f"vLLM ready in {time.time() - t0:.0f}s")

            def fn(arm_id: str):
                ev = evaluators[arm_id]
                return lambda tid, i, n: ev.evaluate_task(task=swe_tasks[tid], task_index=i, total_tasks=n)

            summaries = asyncio.run(run_matrix(
                {a: (fn(a), args.out / a) for a in members}, task_ids, concurrency=args.concurrency,
                repo_of=repo_of, deadline=deadline, sync_in_thread=args.mode == "threads", log=log))
            for a, s in summaries.items():
                log(f"arm {a}: {s['passed']}/{s['tasks_with_rows']} passed, {s['tasks_skipped_deadline']} skipped, "
                    f"wall {s['wall_sec']:.0f}s, effective parallelism {s['effective_parallelism']}")
        finally:
            try:
                server.stop()
            except Exception as exc:  # noqa: BLE001
                log(f"vLLM stop warning: {exc}")
            time.sleep(20 if gi < len(groups) else 0)  # let the GPUs free up before the next server

    # 4. Scorecard.
    card = scorecard([args.out / a for a in arm_ids], control=args.control or arm_ids[0])
    (args.out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(card)
    (args.out / "scorecard.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return 0


class _Logger:
    """The tiny logger interface `_start_server_failing_fast` uses."""

    def error(self, msg: str, *a: Any) -> None:
        log("ERROR " + (msg % a if a else msg))


if __name__ == "__main__":
    sys.exit(main())
