# /// script
# requires-python = ">=3.12"
# dependencies = ["kaggle>=1.7", "wandb>=0.17"]
# ///
"""Push a pipeline notebook to Kaggle, run it, and bring its outputs and logs home.

    uv run scripts/kaggle_push.py -multi_lora_sft                # 31B QLoRA on 4xL4 (PyTorch), full pipeline
    uv run scripts/kaggle_push.py -tpu                           # E4B on a TPU VM (Tunix / JAX), full pipeline
    uv run scripts/kaggle_push.py -tpu --hours 4                 # whole notebook limited to 4 h (clock_hours)
    uv run scripts/kaggle_push.py -tpu --set probe=true          # 2-minute on-TPU check of the Tunix path, then stop
    uv run scripts/kaggle_push.py -training --internet on        # this push with internet (live W&B charts; not on 4xL4)
    uv run scripts/kaggle_push.py -tpu --set tune=true           # Optuna search on the TPU instead of a training run
    uv run scripts/kaggle_push.py -training                      # push, wait for the run, download outputs + log
    uv run scripts/kaggle_push.py -training --set dry_run=true   # RUN overrides baked into the pushed notebook
    uv run scripts/kaggle_push.py -training --no-wait            # push and return
    uv run scripts/kaggle_push.py -training -status              # one status check
    uv run scripts/kaggle_push.py -training -output              # download outputs + log now
    uv run scripts/kaggle_push.py -tpu -output --publish-bundle  # ... and upload the checkpoint bundle as a dataset version

Workflow: commit -> git push -> update the banwait13/gemmafinetuning dataset -> this script. The notebook imports the
repo from that dataset, so this script only stamps the run with the local git SHA and warns when the working tree or
unpushed commits would make the run differ from it.

Pipelines are declared in scripts/kaggle_settings.json; each key under "pipelines" becomes a -<name> flag. Runs are
saved to runs/kaggle/<slug>_<timestamp>/ (gitignored). Credentials: ~/.kaggle/kaggle.json, KAGGLE_* env vars, or the
gitignored repo .env.

Internet: Kaggle does not allow internet on the 4xL4 machine (NvidiaL4), so those pushes are always offline; a settings
file or --internet on that asks for it is refused before anything is pushed. The pushed notebook is told
(RUN["internet"]) so it skips every network step instead of waiting on it.

W&B: a run without internet logs offline into its output. After the download, every `wandb/offline-run-*` folder is
uploaded with `wandb sync` when a W&B key is available here (WANDB_API_KEY in the environment / .env, or a prior
`wandb login`), so offline GPU runs still end up as normal charts in the W&B project. --no-wandb-sync skips it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETTINGS = REPO / "scripts" / "kaggle_settings.json"
PUSH_DIR = REPO / "kaggle" / ".push"
RUNS_DIR = REPO / "runs" / "kaggle"
MARKER = "# @@RUN_OVERRIDES@@"
TERMINAL = {"complete", "error", "cancelrequested", "cancelacknowledged"}
NO_INTERNET_SHAPES = {"NvidiaL4"}  # Kaggle runs the 4xL4 machine without internet; enabling it is not allowed
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


def verify_attached(kid: str, meta: dict, tries: int = 3) -> None:
    """Read the pushed kernel's metadata back: Kaggle can accept a push and silently leave an input off (it dropped
    google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/1 while keeping the w4a16-ct model), and the run
    then dies in cell 3 with 'No base model matching ...'. Slugs are compared case-insensitively (Kaggle rewrites
    'other' -> 'Other')."""
    want = {k: {x.lower() for x in meta[k]} for k in ("dataset_sources", "model_sources")}
    pulled = PUSH_DIR / "_pulled"
    for attempt in range(tries):
        shutil.rmtree(pulled, ignore_errors=True)
        r = kaggle("kernels", "pull", kid, "-p", str(pulled), "-m", capture=True)
        got_file = pulled / "kernel-metadata.json"
        if r.returncode or not got_file.is_file():
            print(f"note: could not read back {kid}'s metadata to check its inputs: {(r.stdout + r.stderr).strip()[-200:]}")
            return
        got = json.loads(got_file.read_text(encoding="utf-8"))
        missing = {k: sorted(want[k] - {x.lower() for x in got.get(k, [])}) for k in want}
        if not any(missing.values()):
            print(f"inputs attached: {len(meta['model_sources'])} model(s), {len(meta['dataset_sources'])} dataset(s)")
            return
        if attempt + 1 < tries:
            time.sleep(10)  # the new version's metadata can lag the push by a few seconds
    sys.exit(f"Kaggle dropped inputs from the pushed kernel: {missing}. The run would fail when it looks for them; "
             f"cancel it at https://www.kaggle.com/code/{kid} (or let it fail in cell 3), check the slug/version with "
             "`kaggle models instances versions files <slug>`, fix scripts/kaggle_settings.json and push again.")


def publicly_readable(api_path: str) -> bool:
    """True when Kaggle serves `api_path` to a request with no credentials, i.e. anyone can read it.

    The owner's own view is no use here: `kernels pull -m` reported is_private: true for a kernel whose source and
    outputs anonymous requests could download (2026-10-31)."""
    try:
        with urllib.request.urlopen(f"https://www.kaggle.com/api/v1/{api_path}", timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False
    except OSError as exc:
        print(f"note: could not check whether {api_path} is public: {exc}")
        return False


def check_private_datasets(ids: list[str]) -> None:
    """Refuse to push when a dataset meant to stay private (mined data, LoRA bundle) is readable by anyone."""
    public = [d for d in ids if publicly_readable(f"datasets/view/{d}")]
    if public:
        sys.exit(f"these datasets must stay private but anyone can read them: {public}. Make them private on "
                 "kaggle.com (dataset -> Settings -> Sharing) and push again.")


def warn_if_kernel_public(kid: str) -> None:
    """A public kernel publishes its outputs: the trained adapters, submission.zip and logs. Loud, not fatal: the run
    has already started, and flipping it to private before it finishes is enough."""
    user, slug = kid.split("/", 1)
    if publicly_readable(f"kernels/pull?userName={user}&kernelSlug={slug}"):
        print(f"\n!!! {kid} is PUBLIC: anyone can download this run's outputs (LoRA adapters, submission.zip, logs).\n"
              f"!!! Make it private before the run finishes: https://www.kaggle.com/code/{kid} -> Settings -> Sharing\n")


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


def existing_datasets(ids: list[str]) -> list[str]:
    """The subset of `ids` that exist and are readable (an optional dataset is simply left off until it has been created)."""
    keep = []
    for d in ids:
        if kaggle("datasets", "files", d, "--page-size", "1", capture=True).returncode == 0:
            keep.append(d)
        else:
            print(f"note: optional dataset {d} does not exist yet; not attaching it (it is created from the first run's bundle)")
    return keep


def wandb_sync(out: Path) -> None:
    """Upload the offline W&B runs found in a downloaded output. Never fails the command."""
    runs = sorted(p for p in out.rglob("offline-run-*") if p.is_dir())
    if not runs:
        return
    netrc = Path.home() / (".netrc" if os.name != "nt" else "_netrc")
    has_key = bool(os.environ.get("WANDB_API_KEY")) or (netrc.is_file() and "api.wandb.ai" in netrc.read_text(errors="ignore"))
    if not has_key:
        print(f"{len(runs)} offline W&B run(s) in {out}; set WANDB_API_KEY (or `wandb login`) and re-run with -output to upload")
        return
    exe = Path(sys.executable).parent / ("wandb.exe" if os.name == "nt" else "wandb")
    cmd = [str(exe) if exe.exists() else "wandb", "sync", *map(str, runs)]
    print(f"uploading {len(runs)} offline W&B run(s): " + " ".join(cmd[:2]) + " ...", flush=True)
    try:
        r = subprocess.run(cmd, check=False, env=UTF8_ENV, timeout=1800)
        print("W&B sync " + ("done: the charts are in your W&B project" if r.returncode == 0 else f"failed ({r.returncode})"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"W&B sync failed: {exc}")


def resolve_internet(p: dict, flag: bool | None) -> bool:
    """The push's enable_internet: --internet wins over the settings; refused on machines that must run offline."""
    wanted = p.get("enable_internet", True) if flag is None else flag
    if wanted and p["machine_shape"] in NO_INTERNET_SHAPES:
        source = "--internet on" if flag else "enable_internet: true in scripts/kaggle_settings.json"
        sys.exit(f"{p['machine_shape']} (4xL4) does not allow internet, but {source} asks for it. Push without it: "
                 "the run logs W&B offline and this script uploads it with `wandb sync` after the download.")
    return wanted


def build_kernel(s: dict, name: str, overrides: dict, internet: bool | None = None) -> tuple[Path, dict]:
    """Copy the notebook into kaggle/.push/<name> with RUN_OVERRIDES filled in, plus its kernel-metadata.json."""
    p = s["pipelines"][name]
    internet = resolve_internet(p, internet)
    overrides = {**overrides, "internet": internet}  # the notebook skips its network steps when False
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
        "enable_internet": internet,
        "competition_sources": [s["competition"]],
        "dataset_sources": [s["repo_dataset"], *p.get("wheelhouses", []), *p.get("private_datasets", []),
                            *existing_datasets(p.get("optional_datasets", []))],
        "kernel_sources": [],
        "model_sources": p.get("model_sources", []),
        "machine_shape": p["machine_shape"],
    }
    if p.get("docker_image"):  # pins the Python/CUDA image; the competition wheelhouse is cp312-only
        meta["docker_image"] = p["docker_image"]
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


def publish_bundle(out: Path, message: str) -> None:
    """Upload the run's checkpoint bundle (bundle.json + dataset-metadata.json) as a version of its private dataset.

    Only runs on an explicit --publish-bundle. Creates the dataset the first time. The bundle was already scanned for
    secrets by the pipeline; the dataset stays private."""
    metas = sorted(out.rglob("dataset-metadata.json"))
    bundles = [m.parent for m in metas if (m.parent / "bundle.json").is_file()]
    if not bundles:
        sys.exit(f"no checkpoint bundle (bundle.json + dataset-metadata.json) under {out}; was output.dataset_id set for the run?")
    bdir = bundles[0]
    ds_id = json.loads((bdir / "dataset-metadata.json").read_text(encoding="utf-8")).get("id")
    if not ds_id or "/" not in ds_id:
        sys.exit(f"{bdir / 'dataset-metadata.json'} has no usable dataset id: {ds_id!r}")
    print(f"publishing {bdir} -> {ds_id}")
    exists = kaggle("datasets", "files", ds_id, "--page-size", "1", capture=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(bdir), "-m", message, "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(bdir), "--dir-mode", "zip")  # private by default
    if publicly_readable(f"datasets/view/{ds_id}"):
        print(f"\n!!! {ds_id} is PUBLIC: anyone can download the LoRA adapters. Make it private now: "
              f"https://www.kaggle.com/datasets/{ds_id} -> Settings -> Sharing\n")
    print(f"https://www.kaggle.com/datasets/{ds_id}  (delete older versions from the dataset page when you no longer need them)")


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
    ap.add_argument("--publish-bundle", action="store_true",
                    help="after downloading outputs, upload the checkpoint bundle as a new dataset version (explicit opt-in)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status checks (default 60)")
    ap.add_argument("--hours", type=float, default=None, metavar="H",
                    help="whole-notebook wall clock in hours, e.g. 4 or 11.3 (sets RUN clock_hours; Kaggle caps 12 GPU / 9 TPU)")
    ap.add_argument("--internet", choices=["on", "off"], default=None,
                    help="override the pipeline's enable_internet for this push (on = live W&B charts; refused on 4xL4)")
    ap.add_argument("--no-wandb-sync", action="store_true", help="do not upload offline W&B runs after downloading outputs")
    args = ap.parse_args()

    p = s["pipelines"][args.pipeline]
    kid = f"{s['username']}/{p['kernel_slug']}"
    if args.action == "status":
        print(f"{kid}: {status(kid)}")
        return
    if args.action == "output":
        out = download(kid, p["kernel_slug"])
        if not args.no_wandb_sync:
            wandb_sync(out)
        if args.publish_bundle:
            publish_bundle(out, f"{p['kernel_slug']} {time.strftime('%Y-%m-%d %H:%M')}")
        return

    if args.hours is not None and not 0.25 <= args.hours <= 12.0:
        sys.exit(f"--hours must be between 0.25 and 12, got {args.hours}")
    clock = {"clock_hours": args.hours} if args.hours is not None else {}
    overrides = {**repo_state(), **p.get("run_defaults", {}), **clock, **parse_sets(args.set)}
    internet = None if args.internet is None else args.internet == "on"
    kdir, meta = build_kernel(s, args.pipeline, overrides, internet)
    overrides["internet"] = meta["enable_internet"]
    print(f"RUN_OVERRIDES = {overrides}  (internet {'on' if meta['enable_internet'] else 'off'})")
    check_sources(meta["dataset_sources"], meta["model_sources"])
    check_private_datasets([*p.get("private_datasets", []), *p.get("optional_datasets", [])])
    kaggle("kernels", "push", "-p", str(kdir))
    print(f"https://www.kaggle.com/code/{kid}")
    verify_attached(kid, meta)
    warn_if_kernel_public(kid)
    if args.no_wait:
        return
    final = wait(kid, args.poll)
    out = download(kid, p["kernel_slug"])
    if not args.no_wandb_sync:
        wandb_sync(out)
    if args.publish_bundle:
        publish_bundle(out, f"{p['kernel_slug']} {time.strftime('%Y-%m-%d %H:%M')}")
    if final != "complete":
        sys.exit(f"run ended with status {final!r}; see run.log above")


if __name__ == "__main__":
    main()
