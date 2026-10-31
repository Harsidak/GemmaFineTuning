"""Phase 6 (`eval`): final-hour vLLM parity check + 3-task `swegemma` smoke evaluation.

Implements design_doc.md §6.2 and §9.1–§9.4:
- Non-fatal to `submission.zip` (which is already built and verified in Phase 5).
- Starts `adk_submission.VllmServer` on staged `runs/<run_id>/submission/`.
- Runs vLLM `/v1/completions` greedy parity check against `parity/<adapter>_hf.json` first (§9.3).
- Runs 3-task smoke eval (1 task in `calibration` mode) via `swegemma.evaluate.Evaluator` with
  `sandbox='subprocess'` and `EventsCompactionConfig`, printing the §9.2 report block and saving `eval/score.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from ..config import PipelineConfig, load_config
from ..runctx import PhaseStatus, RunContext, configure_phase_logging, utc_iso
from ..training.parity import evaluate_parity_against_hf


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction evaluation for incomplete beta via modified Lentz's method."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Computes `I_x(a, b)` in pure Python."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact `(1 - alpha)` Clopper-Pearson confidence interval for `k` successes in `n` trials."""
    if n <= 0:
        return 0.0, 1.0
    k = max(0, min(n, int(k)))

    if k == 0:
        low = 0.0
    else:
        target = alpha / 2.0
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if regularized_incomplete_beta(float(k), float(n - k + 1), mid) < target:
                lo = mid
            else:
                hi = mid
        low = 0.5 * (lo + hi)

    if k == n:
        high = 1.0
    else:
        target = 1.0 - alpha / 2.0
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if regularized_incomplete_beta(float(k + 1), float(n - k), mid) < target:
                lo = mid
            else:
                hi = mid
        high = 0.5 * (lo + hi)

    return low, high


def _format_duration_mmss(sec: float) -> str:
    sec_i = max(0, round(sec))
    m, s = divmod(sec_i, 60)
    return f"{m}m{s:02d}s"


def format_smoke_eval_banner(
    *,
    served_model_id: str,
    adapter_ranks: dict[str, int],
    run_id: str,
    task_rows: list[dict[str, Any]],
    parity_results: dict[str, dict[str, Any]],
) -> str:
    """Formats the exact §9.2 smoke evaluation banner."""
    n = len(task_rows)
    passed = sum(1 for r in task_rows if r.get("result") == "PASS")
    pct = (100.0 * passed / n) if n > 0 else 0.0
    cp_lo, cp_hi = clopper_pearson_interval(passed, n)

    adapters_str = (
        " ".join(f"{name}(r{r})" for name, r in adapter_ranks.items())
        if adapter_ranks
        else "none(bare)"
    )
    lines = [
        f"================ SMOKE EVAL — {n} fixed tasks (NOT a leaderboard estimate) ================",
        f"model: {served_model_id}   adapters: {adapters_str}   run: {run_id}",
        f"{'task':<25} {'result':<8} {'patch_applied':<14} {'turns':<6} {'tool_call_errors':<17} time",
    ]
    for r in task_rows:
        tid = str(r.get("task_id", "unknown"))[:25]
        res = str(r.get("result", "FAIL"))
        pa = "yes" if r.get("patch_applied") is True else ("no" if r.get("patch_applied") is False else "—")
        turns = str(r["turns"]) if r.get("turns") is not None else "—"
        tce = str(r["tool_call_errors"]) if r.get("tool_call_errors") is not None else "—"
        dur = _format_duration_mmss(float(r.get("duration_sec", 0.0)))
        lines.append(f"{tid:<25} {res:<8} {pa:<14} {turns:<6} {tce:<17} {dur}")

    lines.append(
        f"SCORE: {passed}/{n} = {pct:.1f}%   (95% Clopper-Pearson interval: {cp_lo * 100:.1f}%–{cp_hi * 100:.1f}%)"
    )
    if parity_results:
        parts = []
        for aname, pres in parity_results.items():
            parts.append(
                f"{aname} {pres['verdict']} (greedy agreement {pres['adapter_greedy_agreement']:.2f} vs base {pres['base_greedy_agreement']:.2f})"
            )
        lines.append("parity: " + " | ".join(parts))
    else:
        lines.append("parity: skipped")
    lines.append("=" * 90)
    return "\n".join(lines)


def _vllm_completion_greedy(
    *,
    api_base: str,
    model_name: str,
    prompt_text: str,
    max_tokens: int,
    timeout_sec: float = 120.0,
) -> list[dict[str, Any]]:
    """Queries vLLM `/v1/completions` with `temperature=0, logprobs=5` and parses token steps (§9.3)."""
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "logprobs": 5,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    choice = (body.get("choices") or [{}])[0]
    lp_obj = choice.get("logprobs") or {}
    tokens = lp_obj.get("tokens") or []
    token_lps = lp_obj.get("token_logprobs") or []
    top_lps = lp_obj.get("top_logprobs") or []

    steps: list[dict[str, Any]] = []
    for idx, tok in enumerate(tokens):
        steps.append(
            {
                "token": tok,
                "token_id": None,
                "logprob": token_lps[idx] if idx < len(token_lps) else None,
                "top_logprobs": top_lps[idx] if idx < len(top_lps) and isinstance(top_lps[idx], dict) else {},
            }
        )
    return steps


def _discover_staged_adapters(submission_dir: Path) -> tuple[dict[str, Path], dict[str, int]]:
    adapters_dir = submission_dir / "adapters"
    paths: dict[str, Path] = {}
    ranks: dict[str, int] = {}
    if not adapters_dir.is_dir():
        return paths, ranks
    for sub in sorted(adapters_dir.iterdir()):
        if sub.is_dir() and (sub / "adapter_config.json").is_file():
            paths[sub.name] = sub
            cfg = json.loads((sub / "adapter_config.json").read_text(encoding="utf-8"))
            ranks[sub.name] = int(cfg.get("r", 16))
    return paths, ranks


def _extract_task_result_fields(res: Any, task_id: str, duration_sec: float) -> dict[str, Any]:
    """Extracts `result`, `patch_applied`, `turns`, and `tool_call_errors` defensively across swegemma versions."""
    data = res if isinstance(res, dict) else getattr(res, "__dict__", {})

    def _get(*names: str) -> Any:
        for n in names:
            if isinstance(data, dict) and n in data and data[n] is not None:
                return data[n]
            if hasattr(res, n) and getattr(res, n) is not None:
                return getattr(res, n)
        return None

    resolved = bool(_get("resolved", "passed", "success") or False)
    timed_out = bool(_get("timed_out", "timeout") or False)
    err_str = str(_get("error") or "")
    if not timed_out and "timeout" in err_str.lower():
        timed_out = True

    patch_applied = _get("patch_applied")
    if patch_applied is None:
        pred = _get("agent_patch", "prediction", "patch")
        if isinstance(pred, str):
            patch_applied = bool(pred.strip()) and "failed to apply" not in err_str.lower()

    turns = _get("turns", "num_turns", "turn_count")
    tool_errors = _get("tool_call_errors", "tool_errors", "num_tool_errors")
    tool_calls = _get("tool_calls")
    exit_code = _get("test_exit_code")

    return {
        "task_id": task_id,
        "result": "TIMEOUT" if timed_out else ("PASS" if resolved else "FAIL"),
        "resolved": resolved,
        "patch_applied": patch_applied,
        "turns": int(turns) if isinstance(turns, (int, float)) else None,
        "tool_call_errors": int(tool_errors) if isinstance(tool_errors, (int, float)) else None,
        "tool_calls": int(tool_calls) if isinstance(tool_calls, (int, float)) else None,
        "test_exit_code": exit_code,
        "duration_sec": round(duration_sec, 2),
        "error": err_str or None,
    }


# Lines vLLM logs when its engine dies during startup. The api_server parent can stay alive after that, so
# VllmServer.start() (which only checks whether that process exited) waits out the whole startup_timeout.
_FATAL_SERVER_LOG_MARKERS = (
    "Engine core initialization failed",
    "EngineCore failed to start",
    "RuntimeError: Worker failed with error",
)


def _start_server_failing_fast(server: Any, logger: Any, poll_sec: float = 5.0) -> None:
    """Runs `server.start()`, killing the server as soon as its log shows a fatal engine error.

    On 2026-10-31 the 31B engine OOMed 5 minutes into startup and start() then waited 15 more minutes for the
    timeout. Killing the process makes start() see the exit on its next poll and raise with the log tail.
    """
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(poll_sec):
            proc = getattr(server, "process", None)
            log_path = getattr(server, "log_path", None)
            if proc is None or log_path is None or proc.poll() is not None:
                continue
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            marker = next((m for m in _FATAL_SERVER_LOG_MARKERS if m in text), None)
            if marker is not None:
                logger.error("vLLM failed during startup (%r in %s); stopping it instead of waiting", marker, log_path)
                proc.kill()
                return

    watcher = threading.Thread(target=watch, name="vllm-startup-watch", daemon=True)
    watcher.start()
    try:
        server.start()
    finally:
        stop.set()
        watcher.join(timeout=poll_sec + 1.0)


def _build_smoke_evaluator(
    ctx: RunContext,
    cfg: PipelineConfig,
    *,
    model_registry: Any,
    adapter_manifest: Any,
    deadline: float | None,
    notes: list[str],
) -> tuple[Any, dict[str, Any], list[str]]:
    """swegemma Evaluator for the smoke tasks (official "Getting Started" call shape), the task map and the chosen ids."""
    from google.adk.agents.context_cache_config import ContextCacheConfig  # type: ignore[import-not-found]
    from google.adk.apps._configs import EventsCompactionConfig  # type: ignore[import-not-found]
    from swegemma.config import EvalConfig, build_submission_limits  # type: ignore[import-not-found]
    from swegemma.evaluate import Evaluator  # type: ignore[import-not-found]
    from swegemma.models import load_tasks as load_swe_tasks  # type: ignore[import-not-found]

    swe_tasks = {t.instance_id: t for t in load_swe_tasks(Path(cfg.data.tasks_file))}
    target_count = 1 if cfg.run.mode == "calibration" else 3
    chosen_ids = list(cfg.eval.task_ids[:target_count])
    if not chosen_ids:
        # Deterministic tasks from distinct repos when none are configured
        seen_repos: set[str] = set()
        for tid in sorted(swe_tasks):
            if swe_tasks[tid].repo not in seen_repos:
                chosen_ids.append(tid)
                seen_repos.add(swe_tasks[tid].repo)
                if len(chosen_ids) >= target_count:
                    break

    sub_eval: dict[str, Any] = {}
    eval_cfg_yaml = ctx.submission_dir / "eval_config.yaml"
    if eval_cfg_yaml.is_file():
        loaded_ec = yaml.safe_load(eval_cfg_yaml.read_text(encoding="utf-8")) or {}
        sub_eval = loaded_ec.get("evaluation", loaded_ec) if isinstance(loaded_ec, dict) else {}
    max_time_minutes = float(sub_eval.get("max_time_minutes", 60.0))
    if deadline is not None and chosen_ids:
        # Smoke eval only: shrink the per-task agent budget so every chosen task can finish before the phase
        # deadline (~6 min vLLM startup, ~5 min pytest verification per task). The packaged zip keeps its budget.
        fair_share = (deadline - time.time() - 360.0) / 60.0 / len(chosen_ids) - 5.0
        if fair_share < max_time_minutes:
            max_time_minutes = max(2.0, fair_share)
            notes.append(f"Smoke eval agent budget capped at {max_time_minutes:.1f} min/task to fit the deadline")
    turns_raw = sub_eval.get("max_turns", sub_eval.get("max_llm_calls"))
    limits, gen_constraints = build_submission_limits()
    evaluator = Evaluator(
        EvalConfig(
            tasks_path=Path(cfg.data.tasks_file),
            snapshots_dir=Path(cfg.eval.snapshots_dir),
            results_dir=ctx.eval_results_dir,
            submission_dir=ctx.submission_dir,
            models=model_registry,
            sandbox=cfg.eval.sandbox,
            timeout_seconds=int(sub_eval.get("timeout_seconds", 300)),
            verification_timeout_seconds=int(sub_eval.get("verification_timeout_seconds", 300)),
            max_time_minutes=max_time_minutes,
            max_tool_calls=int(sub_eval.get("max_tool_calls", 100)),
            max_turns=int(turns_raw) if turns_raw is not None else None,
            limits=limits,
            generation_constraints=gen_constraints,
            adapter_manifest=adapter_manifest,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(
                compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5
            ),
            graph_dir=cfg.eval.graphs_dir,
            embeddings_dir=cfg.eval.embeddings_dir,
            wheels_dir=Path(cfg.eval.wheels_dir) if cfg.eval.wheels_dir else None,
            verbose=False,
        )
    )
    return evaluator, swe_tasks, chosen_ids


def run_eval_phase(
    ctx: RunContext,
    cfg: PipelineConfig | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Runs the final-hour vLLM parity check when `eval.enabled` is True, plus the 3-task smoke eval when
    `eval.smoke_tasks` is also True."""
    logger = configure_phase_logging(ctx.log_path("eval"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)
    smoke = cfg.eval.smoke_tasks

    skip_reason = None
    if not cfg.eval.enabled:
        skip_reason = "eval.enabled is False"
    elif not smoke and not cfg.parity.enabled:
        skip_reason = "eval.smoke_tasks and parity.enabled are both False: nothing to run"
    if skip_reason is not None:
        t_end = time.time()
        status = PhaseStatus(
            phase="eval",
            state="skipped",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            actual_duration_sec=round(t_end - t_start, 2),
            notes=[skip_reason],
        )
        ctx.write_status(status)
        logger.info("Eval phase skipped (%s)", skip_reason)
        return status.to_dict()

    notes: list[str] = []
    parity_results: dict[str, dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    server = None

    try:
        # The swegemma / adk_submission calls below follow the official "Getting Started - Gemma 4 Developer Agent"
        # notebook (ryanholbrook), the reference for how the scorer drives these packages. Arguments are passed
        # explicitly, so a signature change raises at once instead of being silently dropped.
        import adk_submission  # type: ignore[import-not-found]
        import litellm  # type: ignore[import-not-found]
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS  # type: ignore[import-not-found]

        litellm.drop_params = True
        served_id = cfg.effective_served_model_id()
        served_model_path = cfg.model.served_path or served_id
        if smoke:
            # Only the smoke tasks need the task list and repo snapshots; parity runs without them.
            if not cfg.data.tasks_file or not Path(cfg.data.tasks_file).is_file():
                raise FileNotFoundError(f"data.tasks_file not found: {cfg.data.tasks_file}")
            if not cfg.eval.snapshots_dir or not Path(cfg.eval.snapshots_dir).is_dir():
                raise FileNotFoundError(f"eval.snapshots_dir not found: {cfg.eval.snapshots_dir}")
        else:
            notes.append("Smoke tasks skipped (eval.smoke_tasks=false): vLLM start-up + parity check only")
            logger.info(notes[-1])

        adapter_paths, adapter_ranks = _discover_staged_adapters(ctx.submission_dir)
        # Same discovery the scorer runs: adapters/<name>/ is served as LoRA module <name>, which `adapter: <name>`
        # in the agent YAML routes to. VllmServer mounts these itself (VllmConfig has no lora_modules field).
        adapter_manifest = adk_submission.discover_adapters(
            str(ctx.submission_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS
        )
        served_adapters = set(getattr(adapter_manifest, "adapters", None) or {})
        if served_adapters != set(adapter_paths):
            notes.append(f"Scorer discovers adapters {sorted(served_adapters)}, staged dirs are {sorted(adapter_paths)}")
            logger.warning(notes[-1])

        extra_args: list[str] = []
        if served_id == "gemma-4-12b-it-qat-w4a16-ct":
            extra_args.extend(
                ["--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}']
            )
        if cfg.eval.vllm.max_num_seqs is not None:
            extra_args.extend(["--max-num-seqs", str(cfg.eval.vllm.max_num_seqs)])
        # Count CUDA graph memory inside gpu_memory_utilization (vLLM 0.19 leaves it outside by default, which is
        # what pushed 31B past 22 GiB per L4). VllmServer passes this process's environment to the server.
        os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")

        vllm_cfg = adk_submission.VllmConfig(
            model=served_model_path,
            host=cfg.eval.vllm.host,
            port=cfg.eval.vllm.port,
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
            max_model_len=cfg.eval.vllm.max_model_len,
            dtype="bfloat16",
            gpu_memory_utilization=cfg.eval.vllm.gpu_memory_utilization,
            enable_auto_tool_choice=True,
            enable_lora=bool(served_adapters),
            max_loras=max(1, len(served_adapters)),
            max_lora_rank=max(adapter_ranks.values()) if adapter_ranks else 128,
            scheduler_reserve_full_isl=False,
            tensor_parallel_size=cfg.eval.vllm.tensor_parallel_size,
            startup_timeout=cfg.eval.vllm.startup_timeout,
            extra_args=extra_args,
        )
        server = adk_submission.VllmServer(vllm_cfg, adapter_manifest=adapter_manifest)
        # vLLM serves the base model under its --model path (there is no --served-model-name); the registry maps
        # the YAML's model id onto it and adds "adapter:<name>" aliases for the mounted LoRAs.
        base_model_name = vllm_cfg.model
        model_registry = server.create_model_registry(
            aliases=[served_id], model_prefix="openai/", api_key=os.environ.get("LOCAL_API_KEY", "EMPTY")
        )

        evaluator: Any = None
        swe_tasks: dict[str, Any] = {}
        chosen_ids: list[str] = []
        if smoke:
            # Build the whole evaluator before vLLM loads (5+ min on 31B), so an API mismatch fails in seconds.
            evaluator, swe_tasks, chosen_ids = _build_smoke_evaluator(
                ctx, cfg, model_registry=model_registry, adapter_manifest=adapter_manifest, deadline=deadline,
                notes=notes,
            )
        if hasattr(adk_submission, "install_litellm_reasoning_patch"):
            adk_submission.install_litellm_reasoning_patch()

        logger.info("Starting VllmServer for %s (adapters=%s)", served_id, sorted(served_adapters))
        _start_server_failing_fast(server, logger)
        api_base = f"http://{cfg.eval.vllm.host}:{cfg.eval.vllm.port}/v1"

        # 1. Parity check (§9.3 — runs first before tasks)
        if cfg.parity.enabled:
            # Map trained adapter -> served slot name
            inv_map: dict[str, str] = {}
            for slot_n, target_n in cfg.submission.adapter_map.items():
                if slot_n in served_adapters and target_n not in inv_map:
                    inv_map[target_n] = slot_n

            for spec in cfg.adapters:
                hf_ref_path = ctx.parity_dir / f"{spec.name}_hf.json"
                served_lora_name = inv_map.get(spec.name, spec.name)
                if not hf_ref_path.is_file() or served_lora_name not in served_adapters:
                    continue
                hf_ref = json.loads(hf_ref_path.read_text(encoding="utf-8"))
                gen_toks = int(hf_ref.get("gen_tokens", cfg.parity.gen_tokens))
                v_with: list[list[dict[str, Any]]] = []
                v_base: list[list[dict[str, Any]]] = []
                for p in hf_ref.get("prompts", []):
                    v_with.append(
                        _vllm_completion_greedy(
                            api_base=api_base,
                            model_name=served_lora_name,
                            prompt_text=p["prompt_text"],
                            max_tokens=gen_toks,
                        )
                    )
                    v_base.append(
                        _vllm_completion_greedy(
                            api_base=api_base,
                            model_name=base_model_name,
                            prompt_text=p["prompt_text"],
                            max_tokens=gen_toks,
                        )
                    )
                pres = evaluate_parity_against_hf(
                    hf_ref,
                    v_with,
                    v_base,
                    max_agreement_drop=cfg.parity.max_agreement_drop,
                )
                parity_results[spec.name] = pres
                (ctx.parity_dir / f"{spec.name}_vllm.json").write_text(
                    json.dumps(pres, indent=2), encoding="utf-8"
                )
                if pres["verdict"] == "SUSPECT":
                    notes.append(f"PARITY SUSPECT on {spec.name}: {pres['recommendation']}")
                    logger.warning("PARITY SUSPECT on %s: %s", spec.name, pres)

        # 2. Smoke evaluation on 3 fixed tasks (1 task in calibration mode) (§9.2)
        for idx, tid in enumerate(chosen_ids, start=1):
            if deadline is not None and time.time() >= deadline:
                task_rows.append(
                    {
                        "task_id": tid,
                        "result": "TIMEOUT",
                        "resolved": False,
                        "patch_applied": False,
                        "turns": None,
                        "tool_call_errors": None,
                        "duration_sec": 0.0,
                        "error": "Eval phase deadline reached before task started",
                    }
                )
                continue

            t0_task = time.time()
            try:
                if tid not in swe_tasks:
                    raise KeyError(f"task {tid!r} is not in {cfg.data.tasks_file}")
                logger.info("[%d/%d] Evaluating %s (%s)", idx, len(chosen_ids), tid, swe_tasks[tid].repo)
                res = evaluator.evaluate_task(task=swe_tasks[tid], task_index=idx, total_tasks=len(chosen_ids))
                if asyncio.iscoroutine(res):
                    res = asyncio.run(res)
                row = _extract_task_result_fields(res, tid, time.time() - t0_task)
            except Exception as task_exc:
                logger.exception("Smoke task %s failed", tid)
                row = {
                    "task_id": tid,
                    "result": "FAIL",
                    "resolved": False,
                    "patch_applied": False,
                    "turns": None,
                    "tool_call_errors": None,
                    "duration_sec": round(time.time() - t0_task, 2),
                    "error": f"{type(task_exc).__name__}: {task_exc}",
                }
            logger.info("  -> %s", row)
            task_rows.append(row)

        banner = format_smoke_eval_banner(
            served_model_id=served_id,
            adapter_ranks=adapter_ranks,
            run_id=ctx.run_id,
            task_rows=task_rows,
            parity_results=parity_results,
        )
        print(banner, flush=True)
        logger.info("\n%s", banner)

        passed_cnt = sum(1 for r in task_rows if r["result"] == "PASS")
        n_tasks = len(task_rows)
        cp_lo, cp_hi = clopper_pearson_interval(passed_cnt, n_tasks)
        score_payload = {
            "disclaimer": "Smoke eval verifies adapter loading, end-to-end execution, and patch application; a 3-task sample cannot rank models.",
            "served_model_id": served_id,
            "smoke_tasks": smoke,
            "tasks_evaluated": n_tasks,
            "tasks_passed": passed_cnt,
            "pass_rate": round(passed_cnt / max(1, n_tasks), 4),
            "clopper_pearson_95": [round(cp_lo, 4), round(cp_hi, 4)],
            "tasks": task_rows,
            "parity": parity_results,
            "banner": banner,
        }
        ctx.eval_score_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.eval_score_path.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
        ctx.update_manifest({"eval_score": score_payload, "parity": parity_results})

        t_end = time.time()
        status = PhaseStatus(
            phase="eval",
            state="ok",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            outputs=score_payload,
            notes=notes,
        )
        ctx.write_status(status)
        return status.to_dict()

    except Exception as exc:
        t_end = time.time()
        logger.exception("Eval phase failed (submission.zip unaffected)")
        status = PhaseStatus(
            phase="eval",
            state="failed",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            error=str(exc),
            outputs={"parity": parity_results, "tasks": task_rows},
            notes=notes,
        )
        ctx.write_status(status)
        raise
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception as stop_exc:  # noqa: BLE001
                logger.debug("VllmServer stop warning: %s", stop_exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 6: eval")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    try:
        run_eval_phase(ctx, deadline=args.deadline)
        return 0
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
