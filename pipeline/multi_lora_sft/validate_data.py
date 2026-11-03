"""Checks training-trajectory JSONL files against the pipeline's data contract, before you spend a GPU session on them.

    uv run python scripts/validate_dataset.py data/my_trajectories/*.jsonl
    uv run python scripts/validate_dataset.py --config pipeline/multi_lora_sft/configs/31b_full.yaml --tasks <tasks.jsonl> f.jsonl
    uv run python pipeline/main.py --action validate-data --preset 31b_4x_l4 --train-data f.jsonl

It reports every problem it finds (not just the first), what each adapter would train on, and runs the same
benchmark-leakage gate as the pipeline's preflight against the competition `tasks.jsonl` when one is found. No GPU,
model or tokenizer is needed. The record format is described in README.md ("Training data format").

Exit code: 0 = the pipeline will accept the data, 1 = errors (fix them first), 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.training.data import check_leakage
from pipeline.src.agent.tools import OFFICIAL_HOST_TOOLS

VALID_ROLES = {"user", "assistant", "model", "tool", "system", "developer"}
ASSISTANT_ROLES = {"assistant", "model"}
MAX_EXAMPLES = 5  # per problem kind, so one systematic bug does not flood the output


class Findings:
    def __init__(self) -> None:
        self.errors: dict[str, list[str]] = {}
        self.warnings: dict[str, list[str]] = {}

    def error(self, kind: str, where: str) -> None:
        self.errors.setdefault(kind, []).append(where)

    def warn(self, kind: str, where: str) -> None:
        self.warnings.setdefault(kind, []).append(where)


def _parse_args(raw: Any) -> tuple[bool, Any]:
    if isinstance(raw, dict):
        return True, raw
    if isinstance(raw, str):
        try:
            return True, json.loads(raw)
        except json.JSONDecodeError:
            return False, None
    return False, None


def check_record(rec: Any, where: str, f: Findings, *, need_agent: bool = True) -> dict[str, Any] | None:
    """Validates one record; returns it when the pipeline can load it, else None."""
    if not isinstance(rec, dict):
        f.error("record is not a JSON object", where)
        return None
    tid = rec.get("task_id") or rec.get("instance_id")
    if not isinstance(tid, str) or not tid.strip():
        f.error("missing string 'task_id' (or 'instance_id')", where)
        return None
    where = f"{where} ({tid})"
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        f.error("missing non-empty 'messages' list", where)
        return None
    ok = True
    n_assistant = 0
    call_ids: set[str] = set()
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES:
            f.error(f"message role must be one of {sorted(VALID_ROLES)}", f"{where} message[{i}]")
            ok = False
            continue
        role = m["role"]
        if role in ASSISTANT_ROLES:
            n_assistant += 1
            for j, tc in enumerate(m.get("tool_calls") or []):
                fn = (tc or {}).get("function") if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    f.error("tool_call needs {'function': {'name': str, 'arguments': dict|json-str}}",
                            f"{where} message[{i}].tool_calls[{j}]")
                    continue
                if fn["name"] not in OFFICIAL_HOST_TOOLS:
                    f.warn(f"tool '{fn['name']}' is not one of the 9 harness tools", f"{where} message[{i}]")
                parsed, _ = _parse_args(fn.get("arguments", {}))
                if not parsed:
                    f.error("tool_call arguments are not a JSON object", f"{where} message[{i}].tool_calls[{j}]")
                if isinstance(tc.get("id"), str):
                    call_ids.add(tc["id"])
            if not (m.get("content") or m.get("tool_calls") or m.get("reasoning")):
                f.warn("empty assistant turn (no content, reasoning or tool_calls)", f"{where} message[{i}]")
        elif role == "tool":
            if not m.get("name"):
                f.warn("tool message without 'name'", f"{where} message[{i}]")
            if call_ids and m.get("tool_call_id") and m["tool_call_id"] not in call_ids:
                f.warn("tool_call_id does not match an earlier tool_call id", f"{where} message[{i}]")
    if n_assistant == 0:
        f.error("no assistant/model turn: nothing to train on", where)
        ok = False
    if need_agent and not rec.get("agent"):
        f.error("missing 'agent' (sub-agent role, e.g. \"root\"): adapters select records by it", where)
        ok = False
    if "resolved" in rec and not isinstance(rec["resolved"], bool):
        f.warn("'resolved' should be true/false", where)
    return rec if ok else None


def load_tasks(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def find_tasks_file(explicit: str | None, cfg_tasks: str | None) -> Path | None:
    for cand in (explicit, cfg_tasks, REPO_ROOT / "build" / "gemma-4-developer-agent" / "tasks.jsonl"):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="trajectory JSONL files or globs")
    ap.add_argument("--config", default=None, help="pipeline config whose adapters/roles to check against (default: base)")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="config override, as for the pipeline")
    ap.add_argument("--tasks", default=None, help="competition tasks.jsonl for the leakage gate (auto-detected)")
    ap.add_argument("--json", action="store_true", help="print a machine-readable summary")
    args = ap.parse_args(argv)

    files = sorted({p for pat in args.files for p in (glob.glob(pat) or [pat])})
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        print(f"not found: {missing}", file=sys.stderr)
        return 2
    cfg = load_config(args.config, args.sets)

    need_agent = any(spec.roles for spec in cfg.adapters)
    f = Findings()
    good: list[dict[str, Any]] = []
    total = 0
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                total += 1
                where = f"{path}:{line_no}"
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    f.error("invalid JSON line", f"{where} ({exc.msg})")
                    continue
                ok = check_record(rec, where, f, need_agent=need_agent)
                if ok is not None:
                    good.append(ok)

    dup = [t for t, n in Counter(str(r.get("task_id") or r.get("instance_id")) for r in good).items() if n > 1]
    if dup:
        f.warn("task_id appears more than once (kept; the val split is per task_id)", ", ".join(dup[:20]))

    # What each adapter would train on (same filters as prepare_adapter_dataset)
    usable = [r for r in good if not cfg.data.successful_only or r.get("resolved", True)]
    per_adapter: dict[str, int] = {}
    for spec in cfg.adapters:
        roles = set(spec.roles) if spec.roles else None
        n = sum(1 for r in usable if roles is None or str(r.get("agent")) in roles)
        per_adapter[spec.name] = n
    for name, n in per_adapter.items():
        if n == 0:
            spec = next(a for a in cfg.adapters if a.name == name)
            msg = f"adapter '{name}' (roles {spec.roles}) matches 0 records"
            fix = "set 'agent' on the records or change adapters[].roles"
            if any(per_adapter.values()):
                f.warn(msg + ": it is skipped (its slot ships as the bare model)", fix)
            else:
                f.error(msg + ": preflight fails the whole run (no adapter has data)", fix)
    agents = Counter(str(r.get("agent")) for r in good)

    leak: dict[str, Any] | None = None
    tasks_path = find_tasks_file(args.tasks, cfg.data.tasks_file)
    if tasks_path is not None:
        rep = check_leakage(good, load_tasks(tasks_path), cfg.eval.task_ids)
        leak = rep.to_dict()
        if not rep.passed:
            f.error("BENCHMARK LEAKAGE: training data matches held-out competition tasks",
                    json.dumps({k: v[:10] for k, v in leak.items() if isinstance(v, list) and v and k != "same_repo_warnings"}))
        if rep.same_repo_warnings:
            f.warn("same repository as a held-out task (different commit; allowed, worth knowing)",
                   f"{len(rep.same_repo_warnings)} record(s)")
    else:
        f.warn("leakage gate not run", "no tasks.jsonl found (pass --tasks)")

    summary = {
        "files": len(files), "records": total, "loadable": len(good), "train_eligible": len(usable),
        "agents": dict(agents), "per_adapter": per_adapter, "tasks_file": str(tasks_path) if tasks_path else None,
        "errors": {k: len(v) for k, v in f.errors.items()}, "warnings": {k: len(v) for k, v in f.warnings.items()},
        "leakage_passed": None if leak is None else leak["passed"],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{total} record(s) in {len(files)} file(s): {len(good)} loadable, {len(usable)} eligible "
              f"(successful_only={cfg.data.successful_only})")
        print(f"agents: {dict(agents)}  |  per adapter: {per_adapter}")
        print(f"leakage gate: {'not run' if leak is None else ('PASS' if leak['passed'] else 'FAIL')}"
              + (f" against {tasks_path}" if tasks_path else ""))
        for title, bucket in (("ERRORS", f.errors), ("WARNINGS", f.warnings)):
            if not bucket:
                continue
            print(f"\n{title}")
            for kind, wheres in bucket.items():
                print(f"  - {kind}: {len(wheres)}")
                for w in wheres[:MAX_EXAMPLES]:
                    print(f"      {w}")
                if len(wheres) > MAX_EXAMPLES:
                    print(f"      ... {len(wheres) - MAX_EXAMPLES} more")
        print("\nOK: the pipeline will accept this data." if not f.errors else "\nFix the errors above before launching.")
    return 1 if f.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
