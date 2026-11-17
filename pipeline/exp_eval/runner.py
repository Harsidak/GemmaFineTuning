"""Runs one arm's tasks through the harness with N tasks in flight and writes one JSON row per task.

The harness call is injected (`evaluate_fn(task_id, index, total)`), so this module never imports `swegemma` and the
concurrency, resume and deadline logic is unit-tested on a laptop. In the notebook `evaluate_fn` wraps
`swegemma.evaluate.Evaluator.evaluate_task`, which the official harness also drives concurrently (`--concurrency`).

Output layout (`out_dir`):
    rows.jsonl              one line per finished task (append-only, crash-safe, read back for resume)
    patches/<task>.patch    the agent patch the harness extracted (empty file = empty patch)
    run_summary.json        wall clock, effective parallelism, counts
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROW_SCHEMA = 1
_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def select_task_ids(spec: str | list[str], *, all_ids: list[str], dev30: list[str],
                    repo_of: dict[str, str] | None = None) -> list[str]:
    """`dev30`, `all`, `smoke<N>` (N DEV-30 tasks, round-robin over repos so every repo is hit), or explicit ids."""
    if isinstance(spec, list):
        ids = list(dict.fromkeys(spec))
    elif spec == "dev30":
        ids = list(dev30)
    elif spec == "all":
        ids = list(all_ids)
    elif isinstance(spec, str) and spec.startswith("smoke"):
        n = int(spec[len("smoke"):] or 5)
        repo_of = repo_of or {}
        by_repo: dict[str, list[str]] = {}
        for tid in dev30:
            by_repo.setdefault(repo_of.get(tid, tid.rsplit("_", 1)[0]), []).append(tid)
        ids = []
        while len(ids) < min(n, len(dev30)):
            for repo in sorted(by_repo):
                if by_repo[repo] and len(ids) < n:
                    ids.append(by_repo[repo].pop(0))
    else:
        raise ValueError(f"unknown task set {spec!r}: use dev30, all, smoke<N> or a list of ids")
    unknown = [t for t in ids if t not in set(all_ids)]
    if unknown:
        raise ValueError(f"task ids not in tasks.jsonl: {unknown}")
    return ids


def _get(res: Any, *names: str) -> Any:
    data = res if isinstance(res, dict) else getattr(res, "__dict__", {}) or {}
    for n in names:
        if isinstance(data, dict) and data.get(n) is not None:
            return data[n]
        val = getattr(res, n, None)
        if val is not None and not callable(val):
            return val
    return None


def _jsonable_scalars(res: Any, skip: set[str]) -> dict[str, Any]:
    """Scalar fields of the harness result (pydantic model, dataclass or object), for later inspection."""
    if hasattr(res, "model_dump"):
        try:
            data = res.model_dump()
        except Exception:  # noqa: BLE001
            data = {}
    elif isinstance(res, dict):
        data = res
    else:
        data = dict(getattr(res, "__dict__", {}) or {})
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, (bool, int, float)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v if len(v) <= 2000 else v[:2000] + "...[truncated]"
    return out


def patch_files(patch: str) -> list[str]:
    return sorted({m.group(2) for m in _DIFF_FILE_RE.finditer(patch or "")})


def result_to_row(res: Any, *, task_id: str, repo: str | None, duration_sec: float) -> tuple[dict[str, Any], str]:
    """(row, patch_text) from whatever `evaluate_task` returned."""
    patch = _get(res, "agent_patch", "prediction", "patch") or ""
    if not isinstance(patch, str):
        patch = str(patch)
    error = _get(res, "error")
    resolved = bool(_get(res, "resolved", "passed", "success") or False)
    row = {
        "schema": ROW_SCHEMA,
        "task_id": task_id,
        "repo": repo,
        "resolved": resolved,
        "patch_chars": len(patch),
        "patch_files": patch_files(patch),
        "patch_sha256": hashlib.sha256(patch.encode("utf-8", "replace")).hexdigest() if patch else None,
        "tool_calls": _get(res, "tool_calls"),
        "turns": _get(res, "turns", "num_turns"),
        "test_exit_code": _get(res, "test_exit_code"),
        "error": str(error) if error else None,
        "duration_sec": round(duration_sec, 2),
        "harness_fields": _jsonable_scalars(res, skip={"agent_patch", "prediction", "patch"}),
    }
    return row, patch


def read_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut off by a crash; the task simply reruns
    return rows


async def run_matrix(
    arms: dict[str, tuple[Callable[[str, int, int], Any], Path]],
    task_ids: list[str],
    *,
    concurrency: int = 4,
    repo_of: dict[str, str] | None = None,
    deadline: float | None = None,
    sync_in_thread: bool = False,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.time,
) -> dict[str, dict[str, Any]]:
    """Runs every (arm, task) pair with at most `concurrency` in flight across ALL arms.

    `arms` maps arm id -> (evaluate_fn, out_dir). Jobs are queued task-major (t1 for every arm, then t2, ...), so if
    the session dies or hits `deadline`, every arm has run the same tasks and the paired comparison stays fair.
    Tasks already finished in an arm's `rows.jsonl` are skipped (resume). A job not started before `deadline` gets a
    `skipped: deadline` row, which the tally leaves out. Returns {arm_id: run summary}."""
    repo_of = repo_of or {}
    total = len(task_ids)
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()
    finished: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    done: dict[str, set[str]] = {}
    for arm_id, (_, out_dir) in arms.items():
        out_dir = Path(out_dir)
        (out_dir / "patches").mkdir(parents=True, exist_ok=True)
        done[arm_id] = {r["task_id"] for r in read_rows(out_dir / "rows.jsonl") if not r.get("skipped")}
        if done[arm_id]:
            log(f"[{arm_id}] resume: {len(done[arm_id])} task(s) already done")
    t_wall = clock()

    async def write(arm_id: str, row: dict[str, Any]) -> None:
        async with lock:
            with (Path(arms[arm_id][1]) / "rows.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            finished[arm_id].append(row)

    async def one(arm_id: str, tid: str) -> None:
        evaluate_fn, out_dir = arms[arm_id]
        async with sem:
            idx = task_ids.index(tid) + 1
            if deadline is not None and clock() >= deadline:
                await write(arm_id, {"schema": ROW_SCHEMA, "task_id": tid, "repo": repo_of.get(tid),
                                     "skipped": "deadline"})
                log(f"[{arm_id} {idx}/{total}] {tid}: SKIPPED (deadline)")
                return
            t0 = clock()
            try:
                if sync_in_thread:
                    res = await asyncio.to_thread(evaluate_fn, tid, idx, total)
                else:
                    res = evaluate_fn(tid, idx, total)
                if inspect.isawaitable(res):
                    res = await res
                row, patch = result_to_row(res, task_id=tid, repo=repo_of.get(tid), duration_sec=clock() - t0)
            except Exception as exc:  # noqa: BLE001 - one broken task must not end the run
                row = {"schema": ROW_SCHEMA, "task_id": tid, "repo": repo_of.get(tid), "resolved": False,
                       "patch_chars": 0, "patch_files": [], "patch_sha256": None, "tool_calls": None, "turns": None,
                       "test_exit_code": None, "error": f"runner: {type(exc).__name__}: {exc}"[:2000],
                       "duration_sec": round(clock() - t0, 2), "harness_fields": {}}
                patch = ""
            row["started_ts"] = round(t0, 2)
            row["concurrency"] = int(concurrency)
            (Path(out_dir) / "patches" / f"{tid}.patch").write_text(patch, encoding="utf-8")
            await write(arm_id, row)
            status = "PASS" if row["resolved"] else "FAIL"
            err = f" | error: {row['error'][:160]}" if row.get("error") else ""
            log(f"[{arm_id} {idx}/{total}] {tid}: {status} | patch={row['patch_chars']}B | "
                f"calls={row['tool_calls']} | {row['duration_sec']:.0f}s{err}")

    jobs = [one(a, t) for t in task_ids for a in arms if t not in done[a]]
    await asyncio.gather(*jobs)
    wall = clock() - t_wall
    all_ran = [r for rows in finished.values() for r in rows if not r.get("skipped")]
    busy_all = sum(float(r.get("duration_sec") or 0.0) for r in all_ran)
    summaries: dict[str, dict[str, Any]] = {}
    for arm_id, (_, out_dir) in arms.items():
        ran = [r for r in finished[arm_id] if not r.get("skipped")]
        all_rows = read_rows(Path(out_dir) / "rows.jsonl")
        real = {r["task_id"]: r for r in all_rows if not r.get("skipped")}
        summary = {
            "arm_id": arm_id,
            "tasks_requested": total,
            "tasks_ran_now": len(ran),
            "tasks_skipped_deadline": sum(1 for r in finished[arm_id] if r.get("skipped")),
            "tasks_with_rows": len(real),
            "passed": sum(1 for r in real.values() if r.get("resolved")),
            "concurrency": int(concurrency),
            "arms_in_run": len(arms),
            "wall_sec": round(wall, 1),
            "sum_task_sec": round(sum(float(r.get("duration_sec") or 0.0) for r in ran), 1),
            # Over the whole run: ~concurrency when jobs really overlap, ~1 when the harness call blocks the loop.
            "effective_parallelism": round(busy_all / wall, 2) if wall > 0 else None,
        }
        (Path(out_dir) / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries[arm_id] = summary
    return summaries


async def run_tasks(
    evaluate_fn: Callable[[str, int, int], Any],
    task_ids: list[str],
    out_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Single-arm `run_matrix`."""
    out = await run_matrix({"arm": (evaluate_fn, Path(out_dir))}, task_ids, **kwargs)
    return out["arm"]
