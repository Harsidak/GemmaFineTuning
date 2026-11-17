"""Per-task outcome analysis from runner rows + ATIF traces, per-arm summaries and paired arm comparisons.

The harness trace (`trace_<task>.json`, ATIF-v1.7) has no "why did it end" field, so `end_reason` is inferred, in
this order:
    context_overflow   the harness error says the prompt outgrew the 32k context
    harness_error      any other harness error on the row
    no_trace           no trace file for the task
    submitted          submit_patch returned status ok at least once
    call_cap           a tool said the tool-call budget is exhausted, or reported used >= max_tool_calls
    time_cap           a tool said the time budget is exhausted, or the last step is within 20 s of the cap
    turn_cap           LLM calls >= max_turns
    nudged_out         the trace ends with >= 3 harness continuation nudges in a row
    stopped_no_submit  the agent gave a final text answer without ever submitting
    unknown            none of the above

It also collects the numbers the experiments need: EXP-014 (first edit call, wins without submit_patch), EXP-018 W0
(cap hits with an empty patch, repeated identical calls, token-limit nudges) and EXP-005 Q2 (do skill calls count
against max_tool_calls: compares get_status' `tool_calls_used` with the calls counted before it).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .runner import read_rows

GATED_TOOLS = frozenset({"run_command", "read_file", "edit_file", "write_file", "search_similar_code",
                         "get_code_neighbors", "get_code_subgraph"})
FREE_TOOLS = frozenset({"submit_patch", "get_status"})
SKILL_TOOLS = frozenset({"load_skill", "load_skill_resource", "run_skill_script", "list_skills"})
EDIT_TOOLS = frozenset({"edit_file", "write_file"})
CAP_REASONS = ("call_cap", "time_cap", "turn_cap", "nudged_out")
HARNESS_DEFAULT_BUDGET = {"timeout_seconds": 300, "max_tool_calls": 100, "max_time_minutes": 60.0, "max_turns": 500}

_WARN_RE = re.compile(r"Only (\d+) tool call\(s\) remaining \((\d+)/(\d+) used\)")
_SHELL_EDIT_RE = re.compile(r"sed\s+-i|<<\s*['\"]?PY|>\s*[\w./-]+\.py\b|\btee\s+[\w./-]+\.py\b|\.write_text\(")
_SCRATCH_RE = re.compile(r"(?:^|/)(?:repro|reproduce|debug|scratch|tmp|temp)[\w-]*\.py$|(?:^|/)test_[\w-]*\.py$")


def _obs_json(step_or_call: dict[str, Any]) -> dict[str, Any] | None:
    obs = step_or_call.get("observation") or {}
    content = obs.get("content") if isinstance(obs, dict) else None
    if not isinstance(content, str):
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _obs_text(step: dict[str, Any]) -> str:
    obs = step.get("observation") or {}
    return str(obs.get("content", "")) if isinstance(obs, dict) else str(obs)


def _is_error(data: dict[str, Any] | None, text: str) -> bool:
    if data is not None:
        return data.get("status") == "error" or "error_type" in data
    return '"status": "error"' in text or "Traceback (most recent call last)" in text


def _nudge_kind(message: str) -> str | None:
    m = message or ""
    if "cut off" in m or "finished closing" in m:
        return "tool_call_cutoff"
    if "reached the token limit" in m:
        return "token_limit"
    if "Please continue your work" in m:
        return "continue"
    return None


def analyze_trace(trace: dict[str, Any] | None, row: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Everything the experiments read from one task: counts, flags and the inferred end_reason."""
    out: dict[str, Any] = {"task_id": row.get("task_id"), "repo": row.get("repo"),
                           "resolved": bool(row.get("resolved")), "patch_chars": row.get("patch_chars"),
                           "duration_sec": row.get("duration_sec"), "has_trace": trace is not None}
    error = str(row.get("error") or "")
    steps = (trace or {}).get("steps") or []

    calls: list[tuple[str, Any, dict[str, Any] | None, str]] = []  # (name, args, obs json, obs text) in order
    nudges: list[str] = []
    seen_task_prompt = False
    final_text = False
    llm_calls = 0
    last_elapsed = 0.0
    for st in steps:
        extra = st.get("extra") or {}
        last_elapsed = max(last_elapsed, float(extra.get("elapsed_s") or 0.0))
        llm_calls += int(st.get("llm_call_count") or 0)
        if extra.get("event_type") == "final":
            final_text = True
        if st.get("source") == "user":
            if seen_task_prompt or extra.get("event_type") != "task_prompt":
                nudges.append(_nudge_kind(str(st.get("message", ""))) or "other_user")
            seen_task_prompt = True
            continue
        tcs = st.get("tool_calls") or []
        for i, tc in enumerate(tcs):
            # One observation per step: it belongs to the step's (usually only) tool call. With several calls in a
            # step the observation is attributed to the last one.
            last = i == len(tcs) - 1
            calls.append((str(tc.get("function_name")), tc.get("arguments"),
                          _obs_json(st) if last else None, _obs_text(st) if last else ""))
        if tcs:
            nudges.append("|tool")  # a tool call breaks a run of consecutive nudges

    name_counts = Counter(c[0] for c in calls)
    gated = sum(n for k, n in name_counts.items() if k in GATED_TOOLS)
    skill = sum(n for k, n in name_counts.items() if k in SKILL_TOOLS)
    unknown = sum(n for k, n in name_counts.items() if k not in GATED_TOOLS | FREE_TOOLS | SKILL_TOOLS)
    errors = [c for c in calls if _is_error(c[2], c[3])]

    submits = [c[2] for c in calls if c[0] == "submit_patch" and c[2] and c[2].get("status") == "ok"]
    reported_used: int | None = None
    q2_samples: list[dict[str, int]] = []
    budget_texts: list[str] = []
    g_before = s_before = u_before = g_err = s_err = 0
    first_edit_call: int | None = None
    scratch_writes = 0
    for name, args, data, text in calls:
        failed = _is_error(data, text)
        if name == "get_status" and data and isinstance(data.get("tool_calls_used"), int):
            reported_used = data["tool_calls_used"]
            if s_before:
                q2_samples.append({"used": reported_used, "gated": g_before, "gated_failed": g_err,
                                   "skill": s_before, "skill_failed": s_err, "unknown": u_before})
        if name in GATED_TOOLS:
            g_before += 1
            g_err += int(failed)
        elif name in SKILL_TOOLS:
            s_before += 1
            s_err += int(failed)
        elif name not in FREE_TOOLS:
            u_before += 1
        m = _WARN_RE.search(text or "")
        if m:
            reported_used = max(reported_used or 0, int(m.group(2)))
        if data and failed:
            budget_texts.append(" ".join(str(data.get(k, "")) for k in ("error_type", "error_message", "error",
                                                                          "message")))
        a = args if isinstance(args, dict) else {}
        path = str(a.get("filepath") or a.get("file_path") or a.get("path") or "")
        scratch_or_test = bool(_SCRATCH_RE.search(path)) or path.startswith("/tmp") or "/tests/" in f"/{path}"
        if name == "write_file" and path and not path.startswith("/tmp") and _SCRATCH_RE.search(path):
            scratch_writes += 1
        # EXP-014 wants the first edit to SOURCE: scratch scripts, tests and /tmp files do not count.
        cmd = str(a.get("command", ""))
        is_source_edit = (name in EDIT_TOOLS and path and not scratch_or_test) or (
            name == "run_command" and _SHELL_EDIT_RE.search(cmd) and "/tmp" not in cmd)
        if is_source_edit and first_edit_call is None and not failed:
            first_edit_call = g_before

    canon = Counter((c[0], json.dumps(c[1], sort_keys=True, default=str)) for c in calls if c[0] not in FREE_TOOLS)
    max_identical = max(canon.values(), default=0)

    trailing_nudges = 0
    for n in reversed(nudges):
        if n == "|tool":
            break
        trailing_nudges += 1
    nudge_kinds = Counter(n for n in nudges if n != "|tool")

    budget_err = " ".join(budget_texts).lower()
    max_calls = int(budget.get("max_tool_calls") or HARNESS_DEFAULT_BUDGET["max_tool_calls"])
    max_turns = int(budget.get("max_turns") or HARNESS_DEFAULT_BUDGET["max_turns"])
    max_sec = 60.0 * float(budget.get("max_time_minutes") or HARNESS_DEFAULT_BUDGET["max_time_minutes"])
    used = reported_used if reported_used is not None else gated

    if "contextwindowexceeded" in error.lower() or "maximum context length" in error.lower():
        reason = "context_overflow"
    elif error:
        reason = "harness_error"
    elif trace is None:
        reason = "no_trace"
    elif submits:
        reason = "submitted"
    elif ("budget" in budget_err and "tool" in budget_err) or used >= max_calls:
        reason = "call_cap"
    elif ("budget" in budget_err and "time" in budget_err) or last_elapsed >= max_sec - 20:
        reason = "time_cap"
    elif llm_calls >= max_turns:
        reason = "turn_cap"
    elif trailing_nudges >= 3:
        reason = "nudged_out"
    elif final_text:
        reason = "stopped_no_submit"
    else:
        reason = "unknown"

    fm = (trace or {}).get("final_metrics") or {}
    out.update({
        "end_reason": reason,
        "submitted": bool(submits),
        "last_submit_patch_size": submits[-1].get("patch_size") if submits else None,
        "tool_calls_total": len(calls),
        "gated_calls": gated,
        "skill_calls": skill,
        "unknown_tool_calls": unknown,
        "reported_tool_calls_used": reported_used,
        "tool_errors": len(errors),
        "skill_errors": sum(1 for c in errors if c[0] in SKILL_TOOLS),
        "tool_counts": dict(name_counts),
        "llm_calls": llm_calls,
        "elapsed_sec": round(last_elapsed, 1),
        "nudges": dict(nudge_kinds),
        "trailing_nudges": trailing_nudges,
        "first_edit_call": first_edit_call,
        "scratch_writes_in_workspace": scratch_writes,
        "max_identical_calls": max_identical,
        "loop_flag": max_identical >= 3,
        "skill_files_in_patch": any(str(f).startswith("skills/") for f in row.get("patch_files") or []),
        "prompt_tokens": fm.get("total_prompt_tokens"),
        "completion_tokens": fm.get("total_completion_tokens"),
        "q2_samples": q2_samples,
    })
    return out


# ------------------------------------------------------------------------------------------------- loading results


def find_traces(root: Path) -> dict[str, Path]:
    return {p.stem.removeprefix("trace_"): p for p in sorted(Path(root).rglob("trace_*.json"))}


def latest_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last real (non-skipped) row per task; a task that was only ever skipped keeps its skipped row."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        if r.get("skipped") and tid in out and not out[tid].get("skipped"):
            continue
        out[tid] = r
    return out


def load_arm_results(arm_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """(rows by task, budget, manifest) from a runner output dir. Falls back to an older pipeline `score.json`."""
    arm_dir = Path(arm_dir)
    rows = latest_rows(read_rows(arm_dir / "rows.jsonl"))
    if not rows and (arm_dir / "score.json").is_file():
        score = json.loads((arm_dir / "score.json").read_text(encoding="utf-8"))
        for t in score.get("tasks", []):
            rows[t["task_id"]] = {**t, "repo": t.get("repo") or t["task_id"].rsplit("_", 1)[0],
                                  "patch_chars": None if t.get("patch_applied") is None else
                                  (1 if t.get("patch_applied") else 0)}
    manifest_path = arm_dir / "arm_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    budget = {**HARNESS_DEFAULT_BUDGET, **(manifest.get("eval_config") or {})}
    return rows, budget, manifest


def analyze_arm(arm_dir: Path) -> dict[str, Any]:
    rows, budget, manifest = load_arm_results(arm_dir)
    traces = find_traces(arm_dir)
    tasks = []
    for tid, row in sorted(rows.items()):
        if row.get("skipped"):
            continue
        trace = None
        if tid in traces:
            try:
                trace = json.loads(traces[tid].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                trace = None
        tasks.append(analyze_trace(trace, row, budget))
    return {"arm_id": manifest.get("arm_id") or Path(arm_dir).name, "arm_dir": str(arm_dir), "budget": budget,
            "skipped": sorted(t for t, r in rows.items() if r.get("skipped")), "tasks": tasks,
            "summary": summarize(tasks, budget)}


# --------------------------------------------------------------------------------------------------- statistics


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def q2_verdict(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Do skill tool calls count against max_tool_calls? Votes from every get_status seen after a skill call."""
    votes: Counter[str] = Counter()
    for t in tasks:
        for s in t.get("q2_samples") or []:
            # A call that fails argument validation can error before the budget gate, so each hypothesis allows a
            # range: every failed call of that kind may or may not have been counted.
            lo_not = s["gated"] - s["gated_failed"]
            hi_not = s["gated"] + s["unknown"]
            lo_cnt = lo_not + s["skill"] - s["skill_failed"]
            hi_cnt = hi_not + s["skill"]
            fits_not = lo_not <= s["used"] <= hi_not
            fits_cnt = lo_cnt <= s["used"] <= hi_cnt
            votes["counted" if fits_cnt and not fits_not else
                  "not_counted" if fits_not and not fits_cnt else "inconclusive"] += 1
    decided = {k: v for k, v in votes.items() if k != "inconclusive"}
    answer = "no evidence (no get_status call after a skill call)"
    if decided:
        top = max(decided, key=decided.get)
        answer = top if len(decided) == 1 else f"mixed ({dict(votes)})"
    return {"answer": answer, "votes": dict(votes)}


def summarize(tasks: list[dict[str, Any]], budget: dict[str, Any]) -> dict[str, Any]:
    n = len(tasks)
    passed = sum(1 for t in tasks if t["resolved"])
    lo, hi = wilson_interval(passed, n)
    empty = [t for t in tasks if t.get("patch_chars") == 0]
    reasons = Counter(t["end_reason"] for t in tasks)
    durations = [float(t["duration_sec"]) for t in tasks if t.get("duration_sec") is not None]
    gated = [t["gated_calls"] for t in tasks]
    skill_users = [t for t in tasks if t["skill_calls"]]
    edits = [t["first_edit_call"] for t in tasks]
    return {
        "tasks": n,
        "passed": passed,
        "pass_at_1": round(passed / n, 4) if n else None,
        "ci95_wilson": [round(lo, 3), round(hi, 3)],
        "empty_patch": len(empty),
        "end_reasons": dict(reasons),
        "empty_patch_by_end_reason": dict(Counter(t["end_reason"] for t in empty)),
        "resolved_without_submit": sum(1 for t in tasks if t["resolved"] and not t["submitted"]),
        "resolved_by_end_reason": dict(Counter(t["end_reason"] for t in tasks if t["resolved"])),
        "duration_sec": {"mean": round(statistics.fmean(durations), 1) if durations else None,
                         "p50": _pct(durations, 0.5), "p90": _pct(durations, 0.9), "max": max(durations, default=None)},
        "gated_calls": {"mean": round(statistics.fmean(gated), 1) if gated else None, "p90": _pct(gated, 0.9)},
        "llm_calls_mean": round(statistics.fmean([t["llm_calls"] for t in tasks]), 1) if tasks else None,
        "completion_tokens_mean": _mean_or_none([t.get("completion_tokens") for t in tasks]),
        "tool_errors_total": sum(t["tool_errors"] for t in tasks),
        "skills": {
            "tasks_using_skills": len(skill_users),
            "skill_calls_total": sum(t["skill_calls"] for t in tasks),
            "skill_errors_total": sum(t["skill_errors"] for t in tasks),
            "skill_files_in_patch": sum(1 for t in tasks if t["skill_files_in_patch"]),
        },
        "exp014": {
            "edited_by_call_12": sum(1 for e in edits if e is not None and e <= 12),
            "never_edited": sum(1 for e in edits if e is None),
            "scratch_writes_in_workspace": sum(t["scratch_writes_in_workspace"] for t in tasks),
        },
        "w0": {  # EXP-018 W0: does the agent loop / run into caps with nothing to show?
            "cap_hits": sum(reasons[r] for r in CAP_REASONS),
            "cap_hits_empty_patch": sum(1 for t in empty if t["end_reason"] in CAP_REASONS),
            "context_overflow": reasons.get("context_overflow", 0),
            "loop_flag_tasks": sum(1 for t in tasks if t["loop_flag"]),
            "token_limit_nudge_tasks": sum(1 for t in tasks if t["nudges"].get("token_limit")
                                           or t["nudges"].get("tool_call_cutoff")),
        },
        "q2_skill_calls_counted": q2_verdict(tasks),
        "budget": budget,
    }


def _mean_or_none(xs: list[Any]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float))]
    return round(statistics.fmean(vals), 1) if vals else None


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired comparison of two analyzed arms on the tasks both ran."""
    ra = {t["task_id"]: t for t in a["tasks"]}
    rb = {t["task_id"]: t for t in b["tasks"]}
    common = sorted(ra.keys() & rb.keys())
    both = [t for t in common if ra[t]["resolved"] and rb[t]["resolved"]]
    only_a = [t for t in common if ra[t]["resolved"] and not rb[t]["resolved"]]
    only_b = [t for t in common if rb[t]["resolved"] and not ra[t]["resolved"]]
    return {
        "a": a["arm_id"], "b": b["arm_id"], "common_tasks": len(common),
        "a_passed": len(both) + len(only_a), "b_passed": len(both) + len(only_b),
        "delta_b_minus_a": len(only_b) - len(only_a),
        "both": len(both), "only_a": only_a, "only_b": only_b,
        "neither": len(common) - len(both) - len(only_a) - len(only_b),
        "mcnemar_p": round(mcnemar_exact_p(len(only_a), len(only_b)), 4),
        # tracker.md noise rule: <= 3 tasks is noise unless it repeats with a second seed.
        "beyond_noise_rule": abs(len(only_b) - len(only_a)) > 3,
    }


# ------------------------------------------------------------------------------------------------------- report


def scorecard(arm_dirs: list[Path], control: str | None = None) -> dict[str, Any]:
    arms = [analyze_arm(d) for d in arm_dirs]
    ctrl = next((x for x in arms if x["arm_id"] == control), arms[0] if arms else None)
    comps = [compare(ctrl, x) for x in arms if ctrl is not None and x is not ctrl]
    return {"control": ctrl["arm_id"] if ctrl else None, "arms": arms, "comparisons": comps}


def render_markdown(card: dict[str, Any]) -> str:
    lines = ["# Experiment scorecard", "", f"Control: `{card['control']}`", "",
             ("| arm | tasks | pass | pass@1 | 95% CI | empty patch | submitted | call cap | time cap | turn cap | "
             "nudged out | no submit | ctx overflow | error | p50 s | p90 s | calls mean |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in card["arms"]:
        s = arm["summary"]
        r = s["end_reasons"]
        ci = s["ci95_wilson"]
        lines.append(
            f"| `{arm['arm_id']}` | {s['tasks']} | {s['passed']} | {s['pass_at_1']} | {ci[0]:.2f}–{ci[1]:.2f} | "
            f"{s['empty_patch']} | {r.get('submitted', 0)} | {r.get('call_cap', 0)} | {r.get('time_cap', 0)} | "
            f"{r.get('turn_cap', 0)} | {r.get('nudged_out', 0)} | {r.get('stopped_no_submit', 0)} | "
            f"{r.get('context_overflow', 0)} | {r.get('harness_error', 0) + r.get('no_trace', 0)} | "
            f"{s['duration_sec']['p50']} | {s['duration_sec']['p90']} | {s['gated_calls']['mean']} |")
    if card["comparisons"]:
        lines += ["", "## Paired comparisons (vs control)", "",
                  "| arm | common | control pass | arm pass | delta | only control | only arm | McNemar p | > noise (3) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in card["comparisons"]:
            lines.append(f"| `{c['b']}` | {c['common_tasks']} | {c['a_passed']} | {c['b_passed']} | "
                         f"{c['delta_b_minus_a']:+d} | {', '.join(c['only_a']) or '-'} | "
                         f"{', '.join(c['only_b']) or '-'} | {c['mcnemar_p']} | {c['beyond_noise_rule']} |")
    lines += ["", "## Process metrics", ""]
    for arm in card["arms"]:
        s = arm["summary"]
        lines += [f"### `{arm['arm_id']}`", "",
                  f"- budget: {s['budget']}",
                  f"- empty patch by end reason: {s['empty_patch_by_end_reason'] or '-'}",
                  (f"- resolved without submit_patch (EXP-014): {s['resolved_without_submit']}; edited by call 12: "
                  f"{s['exp014']['edited_by_call_12']}/{s['tasks']}; never edited: {s['exp014']['never_edited']}; "
                  f"scratch files written in /workspace: {s['exp014']['scratch_writes_in_workspace']}"),
                  f"- W0 (EXP-018): {s['w0']}",
                  (f"- skills: {s['skills']}; Q2 skill calls counted against budget: "
                  f"{s['q2_skill_calls_counted']['answer']}"),
                  f"- tool errors: {s['tool_errors_total']}; mean completion tokens: {s['completion_tokens_mean']}",
                  ""]
        if arm["skipped"]:
            lines.append(f"- skipped at deadline (excluded): {', '.join(arm['skipped'])}")
            lines.append("")
    return "\n".join(lines) + "\n"
