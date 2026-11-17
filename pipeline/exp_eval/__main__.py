"""CLI for the experiment toolkit (no GPU, no harness needed).

    uv run python -m pipeline.exp_eval list
    uv run python -m pipeline.exp_eval build exp016_a1_skill_first --out runs/arms
    uv run python -m pipeline.exp_eval build --all --out runs/arms
    uv run python -m pipeline.exp_eval tally runs/kaggle/<run>/exp_eval/<label>/* --control ctrl_022 --out scorecard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import ArmError, build_arm, list_arms
from .tally import render_markdown, scorecard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.exp_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the arms under experiments/arms")
    b = sub.add_parser("build", help="build arm submission dirs + zips + manifests")
    b.add_argument("arms", nargs="*")
    b.add_argument("--all", action="store_true")
    b.add_argument("--out", type=Path, default=Path("runs/arms"))
    t = sub.add_parser("tally", help="scorecard from runner output dirs (one per arm)")
    t.add_argument("arm_dirs", nargs="+", type=Path)
    t.add_argument("--control", default=None, help="arm id to compare the others against (default: first)")
    t.add_argument("--out", type=Path, default=None, help="writes <out>.json and <out>.md")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(list_arms()))
        return 0
    if args.cmd == "build":
        ids = list_arms() if args.all else args.arms
        if not ids:
            ap.error("name arms or pass --all")
        for arm_id in ids:
            try:
                built = build_arm(arm_id, args.out / arm_id)
            except ArmError as exc:
                print(f"FAILED {arm_id}: {exc}", file=sys.stderr)
                return 1
            m = built.manifest
            print(f"{arm_id}: {built.zip_path} ({m['zip_bytes']} B, sha256 {m['zip_sha256'][:12]})")
            print(f"  changed vs {m['base']}: {m['changed_vs_base'] or '-'}")
            for w in m["rule_warnings"]:
                print(f"  WARNING: {w}")
        return 0
    card = scorecard([d for d in args.arm_dirs if d.is_dir()], control=args.control)
    md = render_markdown(card)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
        args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
