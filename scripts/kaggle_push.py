# /// script
# requires-python = ">=3.12"
# dependencies = ["kaggle>=1.7"]
# ///
"""Push a pipeline notebook to Kaggle, run it, and bring its outputs and logs home.

    uv run scripts/kaggle_push.py -training                      # push, wait for the run, download outputs + log
    uv run scripts/kaggle_push.py -training --set dry_run=true   # RUN overrides baked into the pushed notebook
    uv run scripts/kaggle_push.py -training --no-wait            # push and return
    uv run scripts/kaggle_push.py -training -status              # one status check
    uv run scripts/kaggle_push.py -training -output              # download outputs + log now

Workflow: commit -> git push -> update the banwait13/gemmafinetuning dataset -> this script. The notebook imports the
repo from that dataset, so this script only stamps the run with the local git SHA and warns when the working tree or
unpushed commits would make the run differ from it.

Pipelines are declared in scripts/kaggle_settings.json; each key under "pipelines" becomes a -<name> flag. Runs are
saved to runs/kaggle/<slug>_<timestamp>/ (gitignored). Credentials: ~/.kaggle/kaggle.json, KAGGLE_* env vars, or the
gitignored repo .env.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETTINGS = REPO / "scripts" / "kaggle_settings.json"
PUSH_DIR = REPO / "kaggle" / ".push"
RUNS_DIR = REPO / "runs" / "kaggle"
MARKER = "# @@RUN_OVERRIDES@@"
TERMINAL = {"complete", "error", "cancelrequested", "cancelacknowledged"}
UTF8_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # the CLI echoes kernel logs; cp1252 chokes


def load_dotenv(path: Path = REPO / ".env") -> None:
    """KEY=value lines from the gitignored repo .env; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=False, cwd=REPO, capture_output=True, text=True).stdout.strip()


def kaggle_exe() -> str:
    """The CLI installed into this script's uv environment, else whatever is on PATH."""
    local = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    exe = str(local) if local.exists() else shutil.which("kaggle")
    if exe is None:
        sys.exit("kaggle CLI not found: run this script with `uv run scripts/kaggle_push.py ...`")
    return exe


def kaggle(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    if not capture:
        print("$ kaggle " + " ".join(args), flush=True)
    r = subprocess.run([kaggle_exe(), *args], check=False, capture_output=capture, text=True, env=UTF8_ENV,
                       encoding="utf-8", errors="replace")
    if not capture and r.returncode:
        sys.exit(f"kaggle {args[0]} {args[1]} failed ({r.returncode})")
    return r


def check_sources(datasets: list[str], models: list[str]) -> None:
    """Refuse to push when an input is unreachable: Kaggle silently drops it and the run fails much later."""
    bad = [d for d in datasets if kaggle("datasets", "files", d, "--page-size", "1", capture=True).returncode]
    bad += [m for m in models if kaggle("models", "instances", "versions", "files", m, capture=True).returncode]
    if bad:
        sys.exit(f"inputs unreachable (deleted, private to someone else, or misspelled), not pushing: {bad}")


def repo_state() -> dict:
    """Local git SHA plus warnings when the pushed dataset cannot match it."""
    state = {"git_sha": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    ahead = git("rev-list", "--count", "@{u}..HEAD")
    if state["dirty"]:
        print("WARNING: uncommitted changes; the Kaggle dataset will not contain them.")
    if ahead.isdigit() and int(ahead):
        print(f"WARNING: {ahead} commit(s) not pushed to GitHub; the Kaggle dataset is behind HEAD.")
    return state


def parse_sets(pairs: list[str]) -> dict:
    """KEY=VALUE pairs; VALUE is read as JSON when it parses (true, 3, null, [..]), else kept as a string."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            sys.exit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def build_kernel(s: dict, name: str, overrides: dict) -> tuple[Path, dict]:
    """Copy the notebook into kaggle/.push/<name> with RUN_OVERRIDES filled in, plus its kernel-metadata.json."""
    p = s["pipelines"][name]
    nb_path = REPO / p["notebook"]
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        lines = "".join(cell["source"]).splitlines(True)
        for i, line in enumerate(lines):
            if MARKER in line:
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = f"RUN_OVERRIDES = {overrides!r}  {MARKER}  (set by scripts/kaggle_push.py){newline}"
                hits += 1
        cell["source"] = lines
    if hits != 1:
        sys.exit(f"{nb_path}: expected exactly one {MARKER!r} line, found {hits}")
    nb.setdefault("metadata", {}).setdefault(
        "kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    nb["metadata"].setdefault("language_info", {"name": "python"})

    kdir = PUSH_DIR / name
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    (kdir / nb_path.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{s['username']}/{p['kernel_slug']}",
        "title": p["kernel_title"],
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": p.get("is_private", True),
        "enable_gpu": not p["machine_shape"].startswith("Tpu"),
        "enable_tpu": p["machine_shape"].startswith("Tpu"),
        "enable_internet": p.get("enable_internet", True),
        "competition_sources": [s["competition"]],
        "dataset_sources": [s["repo_dataset"], *p.get("wheelhouses", []), *p.get("private_datasets", [])],
        "kernel_sources": [],
        "model_sources": p.get("model_sources", []),
        "machine_shape": p["machine_shape"],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return kdir, meta


def status(kid: str) -> str:
    """'running', 'complete', 'error', ... parsed from `kaggle kernels status`."""
    r = kaggle("kernels", "status", kid, capture=True)
    text = (r.stdout + r.stderr).strip()
    if r.returncode or '"' not in text:
        return f"unknown ({text[-200:]})"
    return text.rsplit('"', 2)[-2].split(".")[-1].lower()


def wait(kid: str, poll: int) -> str:
    """Poll until the new version finishes. Right after a push the API can still report the previous version's
    terminal state, so a terminal state only counts once the run was seen active or after a 5-minute grace period."""
    start, last, seen_active = time.time(), None, False
    while True:
        st = status(kid)
        elapsed = int(time.time() - start)
        if st != last:
            print(f"[{elapsed // 60:3d} min] {kid}: {st}", flush=True)
            last = st
        seen_active |= st in {"queued", "running"}
        if st in TERMINAL and (seen_active or elapsed > 300):
            return st
        time.sleep(poll)


def plain_log(raw: Path, dest: Path) -> None:
    """Kaggle's log is a JSON list of {stream_name, time, data}; write the data as a readable text log."""
    try:
        entries = json.loads(raw.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return
    with dest.open("w", encoding="utf-8") as fh:
        for e in entries:
            prefix = "ERR " if e.get("stream_name") == "stderr" else ""
            for line in str(e.get("data", "")).splitlines():
                fh.write(f"[{e.get('time', 0):9.1f}s] {prefix}{line}\n")


def download(kid: str, slug: str) -> Path:
    out = RUNS_DIR / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    kaggle("kernels", "output", kid, "-p", str(out))
    for raw in out.glob("*.log"):
        plain_log(raw, out / "run.log")
    manifest = out / "outputs" / "run_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"run: {m['status']}  ·  git {m.get('git_sha')}  ·  {m['total_seconds'] // 60} min  ·  adapter {m['adapter']}")
    print(f"outputs in {out}")
    return out


def main() -> None:
    load_dotenv()
    s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    which = ap.add_mutually_exclusive_group(required=True)
    for name, p in s["pipelines"].items():
        which.add_argument(f"-{name}", f"--{name}", dest="pipeline", action="store_const", const=name,
                           help=f"{p['notebook']} -> {s['username']}/{p['kernel_slug']}")
    action = ap.add_mutually_exclusive_group()
    action.add_argument("-status", "--status", dest="action", action="store_const", const="status",
                        help="print the run's status and exit")
    action.add_argument("-output", "--output", dest="action", action="store_const", const="output",
                        help="download outputs + log of the latest run and exit")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a RUN key")
    ap.add_argument("--no-wait", action="store_true", help="push and return without waiting")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status checks (default 60)")
    args = ap.parse_args()

    p = s["pipelines"][args.pipeline]
    kid = f"{s['username']}/{p['kernel_slug']}"
    if args.action == "status":
        print(f"{kid}: {status(kid)}")
        return
    if args.action == "output":
        download(kid, p["kernel_slug"])
        return

    overrides = {**repo_state(), **parse_sets(args.set)}
    kdir, meta = build_kernel(s, args.pipeline, overrides)
    print(f"RUN_OVERRIDES = {overrides}")
    check_sources(meta["dataset_sources"], meta["model_sources"])
    kaggle("kernels", "push", "-p", str(kdir))
    print(f"https://www.kaggle.com/code/{kid}")
    if args.no_wait:
        return
    final = wait(kid, args.poll)
    download(kid, p["kernel_slug"])
    if final != "complete":
        sys.exit(f"run ended with status {final!r}; see run.log above")


if __name__ == "__main__":
    main()
