"""Tally: end-reason inference, EXP-014 / W0 / Q2 metrics and paired comparison, on synthetic ATIF traces shaped
like the real ones (`runs/kaggle/.../eval/results/traces/trace_*.json`), plus the real ones when present."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.exp_eval.arms import REPO_ROOT
from pipeline.exp_eval.tally import (
    analyze_arm,
    analyze_trace,
    compare,
    latest_rows,
    mcnemar_exact_p,
    render_markdown,
    scorecard,
)

BUDGET = {"timeout_seconds": 240, "max_tool_calls": 10, "max_time_minutes": 8, "max_turns": 80}


class Trace:
    """Builds an ATIF-v1.7 trace the way the harness writes it: system + task_prompt, then one agent step per call."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = [
            {"step_id": 1, "source": "system", "message": "sys", "extra": {"event_type": "system_instruction",
                                                                         "elapsed_s": 1.0}},
            {"step_id": 2, "source": "user", "message": "task", "extra": {"event_type": "task_prompt",
                                                                        "elapsed_s": 1.0}},
        ]
        self.t = 1.0

    def call(self, name: str, args: dict | None = None, obs: dict | str | None = None, dt: float = 5.0) -> Trace:
        self.t += dt
        content = obs if isinstance(obs, str) else json.dumps(obs if obs is not None else {"status": "ok"})
        self.steps.append({"step_id": len(self.steps) + 1, "source": "agent", "message": "thinking",
                           "llm_call_count": 1, "extra": {"event_type": "thinking", "elapsed_s": self.t},
                           "tool_calls": [{"tool_call_id": "c", "function_name": name, "arguments": args or {}}],
                           "observation": {"content": content}})
        return self

    def nudge(self, text: str) -> Trace:
        self.steps.append({"step_id": len(self.steps) + 1, "source": "user", "message": text,
                           "extra": {"event_type": "user_message", "elapsed_s": self.t}})
        return self

    def final(self, dt: float = 1.0) -> Trace:
        self.t += dt
        self.steps.append({"step_id": len(self.steps) + 1, "source": "agent", "message": "done", "llm_call_count": 1,
                           "extra": {"event_type": "final", "elapsed_s": self.t}})
        return self

    def build(self) -> dict[str, Any]:
        return {"schema_version": "ATIF-v1.7", "steps": self.steps,
                "final_metrics": {"total_prompt_tokens": 100, "total_completion_tokens": 10}}


def row(tid: str = "t", resolved: bool = False, patch: int = 0, error: str | None = None, **kw: Any) -> dict:
    return {"task_id": tid, "repo": "r", "resolved": resolved, "patch_chars": patch, "error": error,
            "duration_sec": 60.0, "patch_files": kw.get("files", []), "tool_calls": kw.get("calls")}


def test_submitted_task_and_first_source_edit_ignores_scratch_files() -> None:
    tr = (Trace().call("write_file", {"filepath": "repro.py", "content": "x"})
          .call("read_file", {"filepath": "pkg/a.py"})
          .call("edit_file", {"filepath": "tests/test_a.py"})
          .call("edit_file", {"filepath": "pkg/a.py"})
          .call("submit_patch", {}, {"status": "ok", "patch_size": 120, "files_changed": 1}).final().build())
    a = analyze_trace(tr, row(resolved=True, patch=120), BUDGET)
    assert a["end_reason"] == "submitted" and a["last_submit_patch_size"] == 120
    assert a["first_edit_call"] == 4  # the 4th budget-gated call; repro.py and the test edit do not count
    assert a["scratch_writes_in_workspace"] == 1
    assert a["gated_calls"] == 4 and a["tool_calls_total"] == 5


def test_call_cap_from_reported_usage_and_from_the_budget_error() -> None:
    tr = Trace()
    for i in range(9):
        tr.call("read_file", {"filepath": f"f{i}.py"})
    tr.call("get_status", {}, {"status": "ok", "tool_calls_used": 10, "max_tool_calls": 10})
    assert analyze_trace(tr.build(), row(), BUDGET)["end_reason"] == "call_cap"

    tr2 = Trace().call("read_file", {"filepath": "a.py"}).call(
        "run_command", {"command": "ls"},
        {"status": "error", "error_type": "BudgetExceeded", "error_message": "Tool call budget exhausted (10/10)"})
    assert analyze_trace(tr2.build(), row(), BUDGET)["end_reason"] == "call_cap"


def test_time_cap_turn_cap_nudged_out_and_no_submit() -> None:
    slow = Trace().call("run_command", {"command": "pytest"}, dt=8 * 60 - 5).build()
    assert analyze_trace(slow, row(), BUDGET)["end_reason"] == "time_cap"

    turns = Trace()
    for _ in range(80):
        turns.call("get_status", {}, {"status": "ok", "tool_calls_used": 0}, dt=0.1)
    assert analyze_trace(turns.build(), row(), BUDGET)["end_reason"] == "turn_cap"

    nudged = (Trace().call("read_file", {"filepath": "a.py"})
              .nudge("Your previous response reached the token limit while thinking before a tool call was completed.")
              .nudge("Please continue your work using the available tools, or call submit_patch when you have ...")
              .nudge("Please continue your work using the available tools, or call submit_patch when you have ...")
              .build())
    a = analyze_trace(nudged, row(), BUDGET)
    assert a["end_reason"] == "nudged_out" and a["nudges"] == {"token_limit": 1, "continue": 2}

    quit_early = Trace().call("read_file", {"filepath": "a.py"}).final().build()
    assert analyze_trace(quit_early, row(), BUDGET)["end_reason"] == "stopped_no_submit"


def test_errors_on_the_row_win_over_the_trace() -> None:
    tr = Trace().call("read_file", {"filepath": "a.py"}).build()
    msg = "litellm.ContextWindowExceededError: This model's maximum context length is 32768 tokens."
    assert analyze_trace(tr, row(error=msg), BUDGET)["end_reason"] == "context_overflow"
    assert analyze_trace(tr, row(error="Sandbox execution error: boom"), BUDGET)["end_reason"] == "harness_error"
    assert analyze_trace(None, row(), BUDGET)["end_reason"] == "no_trace"


def test_loop_flag_and_skill_files_in_patch() -> None:
    tr = Trace()
    for _ in range(3):
        tr.call("run_command", {"command": "grep -n foo pkg/a.py"})
    a = analyze_trace(tr.build(), row(files=["skills/swe-kit/scripts/locate.py", "pkg/a.py"]), BUDGET)
    assert a["loop_flag"] and a["max_identical_calls"] == 3
    assert a["skill_files_in_patch"]


def _q2_trace(used: int) -> dict:
    return (Trace().call("run_skill_script", {"skill_name": "swe-kit", "file_path": "scripts/locate.py"})
            .call("read_file", {"filepath": "a.py"})
            .call("run_skill_script", {"skill_name": "swe-kit", "file_path": "scripts/read_symbol.py"})
            .call("get_status", {}, {"status": "ok", "tool_calls_used": used}).build())


def test_q2_votes_whether_skill_calls_count_against_the_budget(tmp_path: Path) -> None:
    from pipeline.exp_eval.tally import q2_verdict

    counted = analyze_trace(_q2_trace(3), row("a"), BUDGET)
    not_counted = analyze_trace(_q2_trace(1), row("b"), BUDGET)
    assert q2_verdict([counted])["answer"] == "counted"
    assert q2_verdict([not_counted])["answer"] == "not_counted"
    assert q2_verdict([counted, not_counted])["answer"].startswith("mixed")
    assert q2_verdict([analyze_trace(Trace().build(), row(), BUDGET)])["answer"].startswith("no evidence")


def test_q2_allows_for_calls_that_failed_before_the_budget_gate() -> None:
    from pipeline.exp_eval.tally import q2_verdict

    tr = (Trace().call("read_file", {"filepath": "a.py", "start_line": 9, "end_line": 1},
                       {"status": "error", "error_type": "ValueError", "error_message": "start_line > end_line"})
          .call("run_skill_script", {"skill_name": "swe-kit"})
          .call("get_status", {}, {"status": "ok", "tool_calls_used": 1}).build())
    # used=1: either the skill call counted and the failed read did not, or the reverse -> no vote either way.
    assert q2_verdict([analyze_trace(tr, row(), BUDGET)])["votes"] == {"inconclusive": 1}


def test_paired_comparison_and_mcnemar() -> None:
    def arm(name: str, passes: set[str]) -> dict:
        return {"arm_id": name, "tasks": [{"task_id": t, "resolved": t in passes} for t in "abcdefgh"]}

    c = compare(arm("ctrl", {"a", "b"}), arm("new", {"a", "c", "d", "e", "f", "g"}))
    assert (c["both"], c["only_a"], c["only_b"], c["neither"]) == (1, ["b"], ["c", "d", "e", "f", "g"], 1)
    assert c["delta_b_minus_a"] == 4 and c["beyond_noise_rule"] is True
    assert mcnemar_exact_p(1, 5) == pytest.approx(0.21875)
    assert mcnemar_exact_p(0, 0) == 1.0


def test_latest_rows_keeps_the_real_run_over_an_earlier_deadline_skip() -> None:
    rows = [{"task_id": "a", "skipped": "deadline"}, {"task_id": "a", "resolved": True},
            {"task_id": "b", "resolved": False}, {"task_id": "b", "skipped": "deadline"}]
    out = latest_rows(rows)
    assert out["a"]["resolved"] is True and out["b"].get("skipped") is None


def test_scorecard_end_to_end_from_runner_style_dirs(tmp_path: Path) -> None:
    for arm_id, passes in (("ctrl", {"t1"}), ("new", {"t1", "t2"})):
        d = tmp_path / arm_id
        (d / "harness" / "traces").mkdir(parents=True)
        (d / "arm_manifest.json").write_text(json.dumps({"arm_id": arm_id, "eval_config": BUDGET}))
        with (d / "rows.jsonl").open("w") as fh:
            for tid in ("t1", "t2", "t3"):
                fh.write(json.dumps(row(tid, resolved=tid in passes, patch=50 if tid in passes else 0)) + "\n")
                tr = Trace().call("edit_file", {"filepath": "pkg/a.py"})
                if tid in passes:
                    tr.call("submit_patch", {}, {"status": "ok", "patch_size": 50})
                (d / "harness" / "traces" / f"trace_{tid}.json").write_text(json.dumps(tr.final().build()))
    card = scorecard([tmp_path / "ctrl", tmp_path / "new"], control="ctrl")
    assert card["comparisons"][0]["only_b"] == ["t2"]
    s = card["arms"][1]["summary"]
    assert s["passed"] == 2 and s["empty_patch"] == 1 and s["end_reasons"] == {"submitted": 2, "stopped_no_submit": 1}
    md = render_markdown(card)
    assert "| `new` | 3 | 2 |" in md and "Paired comparisons" in md


REAL = REPO_ROOT / "runs/kaggle/gemmafinetuning-multi-lora-sft_20261031_194344/runs/2026-10-31_134105_31b_full/eval"


@pytest.mark.skipif(not REAL.is_dir(), reason="saved Kaggle traces not present (runs/ is gitignored)")
def test_real_traces_match_the_hand_check() -> None:
    """Hand-checked against results/logs/*.log on 2026-11-17."""
    tasks = {t["task_id"]: t for t in analyze_arm(REAL)["tasks"]}
    assert tasks["fastapi_11194"]["end_reason"] == "submitted" and tasks["fastapi_11194"]["resolved"]
    # 3 repro scripts first; calls 47-48 are edit_file calls rejected for a missing old_string (counted as errors
    # since the 2026-11-18 fix), so the first edit that landed in utils.py is call 49.
    assert tasks["fastapi_11194"]["first_edit_call"] == 49
    assert tasks["fastapi_11194"]["reported_tool_calls_used"] == 53
    assert tasks["httpx_3672"]["end_reason"] == "context_overflow"
    assert tasks["requests_6589"]["end_reason"] == "submitted"
    assert tasks["requests_6589"]["first_edit_call"] == 15
    assert tasks["requests_6589"]["scratch_writes_in_workspace"] == 2


def test_budget_errors_on_the_row_are_caps_not_harness_errors() -> None:
    # 2026-11-17 smoke, fastapi_11194: the row error was the call budget, and the scorecard called it "error".
    tr = Trace().call("read_file", {"filepath": "a.py"}).build()
    assert analyze_trace(tr, row(error="Agent exceeded tool call budget (30 calls)"), BUDGET)["end_reason"] == "call_cap"
    assert analyze_trace(tr, row(error="Agent exceeded time budget (8 minutes)"), BUDGET)["end_reason"] == "time_cap"
    assert analyze_trace(tr, row(error="Agent exceeded max_turns (80)"), BUDGET)["end_reason"] == "turn_cap"
    assert analyze_trace(tr, row(error="Sandbox execution error: boom"), BUDGET)["end_reason"] == "harness_error"
    # The harness's final count on the row also proves the cap when the error text is missing.
    assert analyze_trace(tr, row(calls=10), BUDGET)["end_reason"] == "call_cap"


SKILL_NOT_FOUND = {"error": "Skill '`swe-kit`' not found.", "error_code": "SKILL_NOT_FOUND"}
SKILL_RAN_BUT_FAILED = {"skill_name": "swe-kit", "file_path": "scripts/locate.py",
                        "stdout": json.dumps({"status": "error", "message": "No Python source files found in /x"})}
SKILL_OK = {"skill_name": "swe-kit", "file_path": "scripts/locate.py", "stdout": json.dumps({"status": "ok"})}


def test_skill_errors_count_not_found_and_errors_reported_inside_stdout() -> None:
    # 2026-11-17 smoke: all 16 skill calls failed and the scorecard said skill_errors_total 0.
    tr = (Trace().call("run_skill_script", {"skill_name": "`swe-kit`"}, SKILL_NOT_FOUND)
          .call("run_skill_script", {"skill_name": "swe-kit"}, SKILL_RAN_BUT_FAILED)
          .call("run_skill_script", {"skill_name": "swe-kit"}, SKILL_OK).build())
    a = analyze_trace(tr, row(), BUDGET)
    assert a["skill_calls"] == 3 and a["skill_errors"] == 2 and a["skill_not_run"] == 1
    assert a["tool_errors"] == 2


def test_q2_uses_the_harness_final_count_on_the_row() -> None:
    from pipeline.exp_eval.tally import q2_verdict

    # Shape of rich_2725: gated calls + not-found skill calls (free) + one executed skill call (charged) + submit.
    tr = (Trace().call("run_skill_script", {"skill_name": "`swe-kit`"}, SKILL_NOT_FOUND)
          .call("read_file", {"filepath": "a.py"})
          .call("edit_file", {"filepath": "a.py"})
          .call("run_skill_script", {"skill_name": "swe-kit"}, SKILL_RAN_BUT_FAILED)
          .call("submit_patch", {}, {"status": "ok", "patch_size": 10, "files_changed": 1}).build())
    v = q2_verdict([analyze_trace(tr, row(calls=3, patch=10), BUDGET)])
    assert v["answer"] == "counted" and v["executed_skill_rule_exact"] == "1/1"
    # No skill calls -> still no evidence, even with a count on the row.
    plain = analyze_trace(Trace().call("read_file", {"filepath": "a.py"}).build(), row(calls=1), BUDGET)
    assert q2_verdict([plain])["answer"].startswith("no evidence")


def test_markdown_says_when_the_requested_control_did_not_run(tmp_path: Path) -> None:
    from pipeline.exp_eval.tally import render_markdown as render

    card = {"control": "a1", "control_requested": "ctrl_022", "arms": [], "comparisons": []}
    assert "requested control `ctrl_022` was not in this run" in render(card)
    card["control_requested"] = "a1"
    assert "was not in this run" not in render(card)
