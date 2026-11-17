"""Concurrent task runner against a fake harness: overlap, resume, deadline, crashes and task selection."""

from __future__ import annotations

import asyncio
import json
import time
import types
from pathlib import Path

import pytest

from pipeline.exp_eval.runner import (
    read_rows,
    result_to_row,
    run_tasks,
    select_task_ids,
)


def _fake_evaluator(delay: float, calls: list[str], fail: set[str] | None = None):
    async def evaluate(tid: str, idx: int, total: int):
        calls.append(tid)
        await asyncio.sleep(delay)
        if fail and tid in fail:
            raise RuntimeError(f"sandbox died on {tid}")
        return types.SimpleNamespace(resolved=tid.endswith("1"), agent_patch="diff --git a/pkg/x.py b/pkg/x.py\n+1\n",
                                     tool_calls=7, error=None, test_exit_code=0)
    return evaluate


def test_tasks_really_overlap_and_rows_are_written(tmp_path: Path) -> None:
    calls: list[str] = []
    ids = [f"t{i}" for i in range(8)]
    t0 = time.time()
    summary = asyncio.run(run_tasks(_fake_evaluator(0.2, calls), ids, tmp_path, concurrency=4, log=lambda s: None))
    assert time.time() - t0 < 1.2  # 8 x 0.2 s sequential would be 1.6 s
    assert summary["effective_parallelism"] > 2.5
    rows = read_rows(tmp_path / "rows.jsonl")
    assert sorted(r["task_id"] for r in rows) == ids
    r1 = next(r for r in rows if r["task_id"] == "t1")
    assert r1["resolved"] and r1["patch_files"] == ["pkg/x.py"] and r1["tool_calls"] == 7
    assert (tmp_path / "patches" / "t1.patch").read_text(encoding="utf-8").startswith("diff --git")
    assert json.loads((tmp_path / "run_summary.json").read_text())["passed"] == 1


def test_resume_skips_finished_tasks(tmp_path: Path) -> None:
    calls: list[str] = []
    asyncio.run(run_tasks(_fake_evaluator(0, calls), ["a", "b"], tmp_path, concurrency=2, log=lambda s: None))
    calls.clear()
    asyncio.run(run_tasks(_fake_evaluator(0, calls), ["a", "b", "c"], tmp_path, concurrency=2, log=lambda s: None))
    assert calls == ["c"]
    assert sorted(r["task_id"] for r in read_rows(tmp_path / "rows.jsonl")) == ["a", "b", "c"]


def test_a_crashing_task_becomes_a_failed_row_and_the_rest_run(tmp_path: Path) -> None:
    calls: list[str] = []
    asyncio.run(run_tasks(_fake_evaluator(0, calls, fail={"b"}), ["a", "b", "c"], tmp_path, concurrency=3,
                          log=lambda s: None))
    rows = {r["task_id"]: r for r in read_rows(tmp_path / "rows.jsonl")}
    assert set(rows) == {"a", "b", "c"}
    assert rows["b"]["resolved"] is False and "sandbox died on b" in rows["b"]["error"]
    assert (tmp_path / "patches" / "b.patch").read_text() == ""


def test_tasks_not_started_before_the_deadline_are_skipped(tmp_path: Path) -> None:
    calls: list[str] = []
    asyncio.run(run_tasks(_fake_evaluator(0, calls), ["a", "b"], tmp_path, concurrency=1,
                          deadline=time.time() - 1, log=lambda s: None))
    assert calls == []
    assert all(r["skipped"] == "deadline" for r in read_rows(tmp_path / "rows.jsonl"))
    # A later run with time left picks the skipped tasks up again.
    asyncio.run(run_tasks(_fake_evaluator(0, calls), ["a", "b"], tmp_path, concurrency=1, log=lambda s: None))
    assert sorted(calls) == ["a", "b"]


def test_blocking_harness_call_can_run_in_threads(tmp_path: Path) -> None:
    def blocking(tid: str, idx: int, total: int):
        time.sleep(0.2)
        return {"resolved": True, "agent_patch": ""}

    summary = asyncio.run(run_tasks(blocking, ["a", "b", "c", "d"], tmp_path, concurrency=4, sync_in_thread=True,
                                    log=lambda s: None))
    assert summary["effective_parallelism"] > 2.5
    assert all(r["patch_chars"] == 0 for r in read_rows(tmp_path / "rows.jsonl"))


def test_result_to_row_reads_pydantic_like_results() -> None:
    class Res:
        def model_dump(self):
            return {"resolved": True, "agent_patch": "x" * 10, "tool_calls": 3, "notes": "y" * 3000, "_p": 1}

        resolved = True
        agent_patch = "diff --git a/a.py b/a.py\n"
        tool_calls = 3

    row, _patch = result_to_row(Res(), task_id="t", repo="r", duration_sec=1.234)
    assert row["resolved"] and row["patch_files"] == ["a.py"] and row["duration_sec"] == 1.23
    assert "agent_patch" not in row["harness_fields"] and "_p" not in row["harness_fields"]
    assert row["harness_fields"]["notes"].endswith("[truncated]")


def test_task_selection() -> None:
    dev30 = ["fastapi_1", "fastapi_2", "fastapi_3", "rich_1", "httpx_1", "requests_1", "requests_2"]
    all_ids = dev30 + ["rich_9"]
    assert select_task_ids("smoke5", all_ids=all_ids, dev30=dev30) == [
        "fastapi_1", "httpx_1", "requests_1", "rich_1", "fastapi_2"]
    assert select_task_ids("dev30", all_ids=all_ids, dev30=dev30) == dev30
    assert select_task_ids("all", all_ids=all_ids, dev30=dev30) == all_ids
    assert select_task_ids(["rich_9", "rich_9"], all_ids=all_ids, dev30=dev30) == ["rich_9"]
    with pytest.raises(ValueError, match="not in tasks.jsonl"):
        select_task_ids(["nope_1"], all_ids=all_ids, dev30=dev30)
    with pytest.raises(ValueError, match="unknown task set"):
        select_task_ids("dev31", all_ids=all_ids, dev30=dev30)


def test_matrix_interleaves_arms_task_major_and_shares_the_slots(tmp_path: Path) -> None:
    from pipeline.exp_eval.runner import run_matrix

    order: list[str] = []

    def make(arm: str):
        async def ev(tid: str, idx: int, total: int):
            order.append(f"{arm}:{tid}")
            await asyncio.sleep(0.01)
            return {"resolved": arm == "b", "agent_patch": ""}
        return ev

    arms = {a: (make(a), tmp_path / a) for a in ("a", "b", "c")}
    out = asyncio.run(run_matrix(arms, ["t1", "t2"], concurrency=1, log=lambda s: None))
    assert order == ["a:t1", "b:t1", "c:t1", "a:t2", "b:t2", "c:t2"]
    assert out["b"]["passed"] == 2 and out["a"]["passed"] == 0
    assert all(len(read_rows(tmp_path / a / "rows.jsonl")) == 2 for a in arms)


def test_a_deadline_cut_leaves_every_arm_with_the_same_tasks(tmp_path: Path) -> None:
    from pipeline.exp_eval.runner import run_matrix

    now = [0.0]

    def clock() -> float:
        return now[0]

    def make(arm: str):
        async def ev(tid: str, idx: int, total: int):
            now[0] += 10.0  # each job takes 10 "seconds"
            return {"resolved": True, "agent_patch": ""}
        return ev

    arms = {a: (make(a), tmp_path / a) for a in ("a", "b")}
    asyncio.run(run_matrix(arms, ["t1", "t2", "t3"], concurrency=1, deadline=25.0, clock=clock,
                           log=lambda s: None))
    ran = {a: sorted(r["task_id"] for r in read_rows(tmp_path / a / "rows.jsonl") if not r.get("skipped"))
           for a in arms}
    # 3 jobs fit before t=25 (a:t1, b:t1, a:t2); b:t2 is cut, so the tally's common set is {t1}, never lopsided arms.
    assert ran == {"a": ["t1", "t2"], "b": ["t1"]}
