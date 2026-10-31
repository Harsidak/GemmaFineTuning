
import email
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

OUT = Path("/kaggle/working/train_wheels")
OUT_TPU = Path("/kaggle/working/train_wheels_tpu")  # separate wheelhouse: only installed for the Tunix (JAX/TPU) backend
# Which wheelhouses to build: "core" (PyTorch/TRL stack, GPUs) and/or "tpu" (JAX + Tunix). Override with WHEEL_BUNDLES=tpu.
BUNDLES = {b.strip() for b in os.environ.get("WHEEL_BUNDLES", "core,tpu").split(",") if b.strip()}
SCRATCH = Path("/tmp/wheel_build")
TARGET_PY = tuple(int(x) for x in os.environ.get("WHEEL_PYTHON", "3.12").split("."))  # the official wheelhouse is cp312;
# for a TPU image on another Python run:  WHEEL_BUNDLES=tpu WHEEL_PYTHON=3.13  (check `python --version` in a TPU notebook)

# Tested locally; second set is the fallback if the first cannot be resolved/verified.
TESTED = ["peft==0.21.2", "trl==1.16.0", "accelerate==1.15.0", "datasets==5.1.0"]
RANGES = ["peft>=0.17,<1", "trl>=1.0,<2", "accelerate>=1.10.1,<2", "datasets>=4.7,<6"]
EXTRAS = ["optuna>=3.6,<5", "wandb>=0.17,<1", "matplotlib>=3.5", "pandas>=1.5",
          "regex"]  # transformers imports it; bundle it in case the target image lacks it
CANDIDATES = [("tested-pins", TESTED + EXTRAS), ("ranges", RANGES + EXTRAS)]

# ---- TPU bundle (JAX + Tunix). google-tunix declares ~25 dependencies (tensorflow-datasets, numba, kagglehub, ...) of
# which the trainer imports almost none; installing it with --no-deps plus the import-time set below was verified by
# running tpu_training/tests (fresh run, resume, bad checkpoint, NaN abort) in an environment holding exactly these.
TUNIX_PIN = "google-tunix==0.1.7"
TPU_TESTED = [
    "jax[tpu]==0.12.0.1", "flax==0.12.12", "optax==0.2.8", "qwix==0.1.8", "orbax-checkpoint==0.12.8",
    "google-metrax==0.2.4", "clu==0.0.12", "tensorboardX==2.6.5", "jaxtyping==0.3.11", "sentencepiece==0.2.2",
    "perfetto==0.59.1", "wadler-lindig", "wrapt", "tqdm", "pillow", "pandas", "python-dateutil", "pyyaml", "pyarrow", "fsspec",
]
TPU_RANGES = [
    "jax[tpu]>=0.8,<1", "flax>=0.12,<0.13", "optax>=0.2,<0.3", "qwix>=0.1.5,<0.2", "orbax-checkpoint>=0.11,<1",
    "google-metrax", "clu", "tensorboardX", "jaxtyping", "sentencepiece", "perfetto", "wadler-lindig", "wrapt", "tqdm",
    "pillow", "pandas", "python-dateutil", "pyyaml", "pyarrow", "fsspec",
]
TPU_CANDIDATES = [("tested-pins", TPU_TESTED), ("ranges", TPU_RANGES)]
# Declared by google-tunix but not imported by anything the trainer uses (proved by the smoke test + tpu_training tests).
TPU_WAIVED = ("tensorflow-datasets", "numba", "kagglehub", "omegaconf", "hf-transfer", "sympy", "jinja2", "pylatexenc",
              "python-dotenv", "tenacity", "importlib-resources", "datasets", "transformers", "grain")

# Official wheelhouse contents (Getting Started notebook, 2026-10-30); overridden by the real
# wheelhouse when it is attached to this notebook.
OFFICIAL_FALLBACK = {
    "adk-eval-core": "0.1.0", "adk-submission": "0.2.13", "anthropic": "1.4.0",
    "apache-tvm-ffi": "0.1.13.post3", "astor": "0.8.1", "bitsandbytes": "0.46.1",
    "cbor2": "6.1.4", "compressed-tensors": "0.15.0.1", "depyf": "0.20.0",
    "diskcache": "5.6.3", "flashinfer-cubin": "0.6.6", "flashinfer-python": "0.6.6",
    "gguf": "0.19.0", "google-adk": "1.36.1", "google-genai": "2.11.0", "ijson": "3.5.1",
    "interegular": "0.3.3", "llguidance": "1.3.0", "lm-format-enforcer": "0.11.3",
    "loguru": "0.7.3", "mistral-common": "1.11.7", "msgspec": "0.21.1",
    "nvidia-cudnn-frontend": "1.18.0", "openai-harmony": "0.0.8",
    "opentelemetry-exporter-otlp": "1.44.0", "outlines-core": "0.2.11",
    "partial-json-parser": "0.2.1.1.post7", "pybase64": "1.5.0", "pycountry": "26.2.16",
    "quack-kernels": "0.6.4", "safetensors": "0.8.0", "setproctitle": "1.3.7",
    "swegemma": "0.2.11", "tokenizers": "0.22.2", "transformers": "5.13.1",
    "vllm": "0.19.1", "xgrammar": "0.2.5.post1",
}
BASE_IMAGE = ("torch", "torchvision", "torchaudio", "triton", "numpy")  # provided by the image
PREINSTALL = ("transformers", "tokenizers", "safetensors")  # official libs the smoke test runs on

ENV = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}


class StepError(Exception):
    pass


PY = None  # interpreter used for every venv and `pip wheel`: a real CPython 3.12 (wheel tags must match the target)
UV = [sys.executable, "-m", "uv"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run(cmd: list, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd[1:] if c != "-m"))
    return subprocess.run([str(c) for c in cmd], capture_output=capture, text=True, check=check, env=ENV)


def pip(py, *args: str, **kw) -> subprocess.CompletedProcess:
    return run([py, "-m", "pip", *args], **kw)


def tail(text: str, n: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def wheel_name_version(path: Path) -> tuple:
    parts = path.name.split("-")
    return norm(parts[0]), parts[1]


def find_official_wheels() -> tuple:
    """Return (dirs, {name: version}) of the competition wheelhouse if it is attached."""
    root = Path("/kaggle/input")
    dirs, versions = set(), {}
    if root.exists():
        for whl in root.rglob("*.whl"):
            if "developer-agent" in str(whl.parent).lower():
                dirs.add(whl.parent)
                name, version = wheel_name_version(whl)
                versions[name] = version.replace("cu128", "+cu128") if "+" not in version and "cu128" in version else version
    return sorted(dirs), versions


def installed_versions(names) -> dict:
    out = {}
    for n in names:
        try:
            out[norm(n)] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def ensure_py312() -> str:
    """The notebook's own Python may differ from the target (no ensurepip, wrong wheel tags); fetch the target version with uv."""
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True, env=ENV)
    want = "%d.%d" % TARGET_PY
    run([*UV, "python", "install", want])
    path = subprocess.check_output([*UV, "python", "find", want], text=True, env=ENV).strip()
    log(f"build interpreter: {path}")
    return path


def make_venv(venv: Path) -> Path:
    shutil.rmtree(venv, ignore_errors=True)
    run([*UV, "venv", "--python", PY, "--seed", venv])
    return venv / "bin" / "python"


def make_resolve_venv(fakes: dict) -> Path:
    """Clean venv whose 'installed' set is exactly the packages the target machine provides."""
    py = make_venv(SCRATCH / "resolve_venv")
    site = Path(subprocess.check_output([py, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip())
    for name, version in fakes.items():
        d = site / f"{name.replace('-', '_')}-{version}.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (d / "INSTALLER").write_text("fake\n")
        (d / "RECORD").write_text("")
    return py


def preinstall_requirements(official_versions: dict, fakes: dict) -> list:
    """Runtime requirements of the target-owned HF libs (transformers & co.), as resolver constraints.

    The target installs those with --no-deps, so nothing there supplies their dependencies and, worse,
    an unconstrained solve picks e.g. huggingface-hub 2.x that transformers' import-time version check
    rejects. Folding their real Requires-Dist into the solve fixes both."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    wh = SCRATCH / "official_wh"
    wh.mkdir(parents=True, exist_ok=True)
    env = {"extra": "", "python_version": "%d.%d" % TARGET_PY, "python_full_version": "%d.%d.0" % TARGET_PY}
    out = {}
    for n in PREINSTALL:
        pip(PY, "wheel", "--no-deps", "--quiet", "-w", str(wh), f"{n}=={official_versions.get(n, OFFICIAL_FALLBACK[n])}")
        whl = next(wh.glob(f"{n.replace('-', '_')}-*.whl"))
        for line in read_wheel_meta(whl).get_all("Requires-Dist") or []:
            req = Requirement(line)
            if req.marker and not req.marker.evaluate(env):
                continue
            have = fakes.get(norm(req.name))
            if have is not None:  # target-owned: must already satisfy the constraint
                if not req.specifier.contains(have, prereleases=True):
                    log(f"WARNING: {n} wants {line} but the target provides {req.name} {have}")
                continue
            out.setdefault(norm(req.name), []).append(str(req.specifier))  # several libs may constrain one dep
    merged = sorted(f"{name}{','.join(s for s in specs if s)}" for name, specs in out.items())
    log(f"requirements inherited from {', '.join(PREINSTALL)}: {merged}")
    return merged


def resolve(reqs: list, py: Path, fakes: dict) -> list:
    report_path = SCRATCH / "report.json"
    report_path.unlink(missing_ok=True)
    r = pip(py, "install", "--dry-run", "--quiet", "--report", str(report_path), *reqs, capture=True, check=False)
    if r.returncode:
        raise StepError("resolver failed:\n" + tail(r.stderr + r.stdout))
    bundle, conflicts = [], []
    for item in json.loads(report_path.read_text())["install"]:
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        if norm(name) in fakes:
            conflicts.append(f"{name}: needs {version}, target provides {fakes[norm(name)]}")
        else:
            bundle.append((norm(name), version))
    if conflicts:
        raise StepError("requirement set would replace target-owned packages:\n  " + "\n  ".join(conflicts))
    if not bundle:
        raise StepError("resolver produced an empty bundle")
    return sorted(bundle)


def build_wheels(bundle: list, out: Path = OUT, extra_specs: tuple = ()) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for spec in [*(f"{n}=={v}" for n, v in bundle), *extra_specs]:
        for attempt in range(1, 4):
            r = pip(PY, "wheel", "--no-deps", "--quiet", "-w", str(out), spec, capture=True, check=False)
            if r.returncode == 0:
                break
            log(f"  retry {attempt}/3 for {spec}: {tail(r.stderr, 3)}")
            time.sleep(3 * attempt)
        else:
            raise StepError(f"could not build a wheel for {spec}")
    bad = [w.name for w in out.glob("*.whl") if "+" in w.name]
    if bad:
        log(f"WARNING: local-version wheel names get their '+' stripped by Kaggle uploads: {bad}")


def read_wheel_meta(whl: Path):
    with zipfile.ZipFile(whl) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(z.read(meta))


def closure_check(fakes: dict, out: Path = OUT, waived: frozenset = frozenset()) -> dict:
    """Every requirement of every bundled wheel must be met by the bundle or the target image.

    `waived` names dependencies that are deliberately absent (declared by a --no-deps wheel, proven unneeded by the
    smoke test); they are reported separately instead of failing the check."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    metas = [read_wheel_meta(w) for w in sorted(out.glob("*.whl"))]
    available = dict(fakes)
    available.update({norm(m["Name"]): m["Version"] for m in metas})
    missing, conflicts, waived_hits = [], [], []
    for m in metas:
        for line in m.get_all("Requires-Dist") or []:
            req = Requirement(line)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            have = available.get(norm(req.name))
            if have is None and norm(req.name) in waived:
                waived_hits.append(f"{m['Name']} -> {req.name}")
            elif have is None:
                missing.append(f"{m['Name']} -> {line}")
            elif not req.specifier.contains(have, prereleases=True):
                conflicts.append(f"{m['Name']} -> {line} (have {have})")
    return {"missing": missing, "conflicts": conflicts, "waived": waived_hits}


SMOKE = r'''
import json, os, sys, tempfile, traceback
os.environ.update(WANDB_MODE="offline", WANDB_SILENT="true", WANDB_DIR=tempfile.mkdtemp(),
                  TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
out = {"core": {}, "e2e": None}
try:
    import torch, transformers, tokenizers, safetensors, peft, trl, accelerate, datasets, optuna, wandb
    for m in (torch, transformers, tokenizers, safetensors, peft, trl, accelerate, datasets, optuna, wandb):
        out["core"][m.__name__] = getattr(m, "__version__", "?")
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=32, vocab_size=128, n_positions=64))
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM"))
    out["core"]["lora_trainable_params"] = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    SFTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1, report_to="none", use_cpu=True, bf16=False)
    run = wandb.init(project="smoke", mode="offline", dir=os.environ["WANDB_DIR"])
    run.log({"x": 1}); run.finish()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: (t.suggest_float("lr", 1e-5, 1e-3, log=True) - 1e-4) ** 2, n_trials=3)
    out["core"]["ok"] = True
except Exception:
    out["core"]["ok"] = False
    out["core"]["error"] = traceback.format_exc()[-1500:]
if out["core"].get("ok"):
    try:  # tiny end-to-end SFT step on CPU with W&B offline logging (informational, not fatal)
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        tk = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3, "c": 4}, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]")
        ds = datasets.Dataset.from_dict({"text": ["a b c a b c"] * 8})
        cfg = SFTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=2, logging_steps=1,
                        save_strategy="no", report_to="wandb", use_cpu=True, bf16=False, dataset_text_field="text")
        SFTTrainer(model=pm, args=cfg, train_dataset=ds, processing_class=tok).train()
        out["e2e"] = "ok"
    except Exception:
        out["e2e"] = "FAILED: " + traceback.format_exc()[-1500:]
print("SMOKE_JSON:" + json.dumps(out))
sys.exit(0 if out["core"].get("ok") else 1)
'''


def smoke_test(official_dirs: list, official_versions: dict, bundle_names: set, fakes: dict) -> dict:
    py = make_venv(SCRATCH / "smoke_venv")
    # The target image supplies torch/numpy; stand in with CPU builds of the same versions.
    tv = fakes.get("torch", "").split("+")[0]
    for pin in ([f"torch=={tv}"] if tv else []) + [""]:
        r = pip(py, "install", "--quiet", "--index-url", "https://download.pytorch.org/whl/cpu",
                "--extra-index-url", "https://pypi.org/simple", pin or "torch", f"numpy=={fakes.get('numpy', '')}" if fakes.get("numpy") else "numpy",
                capture=True, check=False)
        if not r.returncode:
            break
        log(f"torch CPU install failed ({pin or 'unpinned'}): {tail(r.stderr, 4)}")
    else:
        raise StepError("could not install CPU torch for the smoke test")
    # Mirror the target: official transformers/tokenizers/safetensors first ...
    wh = SCRATCH / "official_wh"
    wh.mkdir(parents=True, exist_ok=True)
    specs = [f"{n}=={official_versions.get(n, OFFICIAL_FALLBACK[n])}" for n in PREINSTALL]
    links = [str(d) for d in official_dirs] or [str(wh)]
    if not official_dirs:
        for spec in specs:
            pip(PY, "wheel", "--no-deps", "--quiet", "-w", str(wh), spec)
    pip(py, "install", "--quiet", "--no-index", "--no-deps", *[a for l in links for a in ("--find-links", l)], *specs)
    # ... then the bundle, exactly like the training notebook (--no-deps from a private wheelhouse).
    pip(py, "install", "--quiet", "--no-index", "--no-deps", *map(str, sorted(OUT.glob("*.whl"))))
    chk = pip(py, "check", capture=True, check=False)
    ours = []
    for line in chk.stdout.splitlines():
        m = re.match(r"^(\S+) \S+ (?:has requirement|requires) ([A-Za-z0-9_.\-]+)", line)
        if m and (norm(m.group(1)) in bundle_names or norm(m.group(1)) in PREINSTALL):
            ours.append(line)
    log(f"pip check: {len(chk.stdout.splitlines())} lines, {len(ours)} concern the bundle")
    r = run([py, "-I", "-c", SMOKE], capture=True, check=False)
    log(tail(r.stderr, 8))
    line = next((l for l in r.stdout.splitlines() if l.startswith("SMOKE_JSON:")), None)
    if not line:
        raise StepError("smoke test produced no result:\n" + tail(r.stdout + r.stderr))
    result = json.loads(line[len("SMOKE_JSON:"):])
    result["pip_check_bundle_issues"] = ours
    if ours or not result["core"].get("ok"):
        raise StepError("smoke test failed:\n" + json.dumps(result, indent=1)[:2500])
    return result


TPU_SMOKE = r"""
import importlib.util, json, os, sys, traceback
os.environ.update(JAX_PLATFORMS="cpu", TF_CPP_MIN_LOG_LEVEL="3")
out = {"ok": False}
try:
    import jax, jax.numpy as jnp, flax, optax, qwix, orbax.checkpoint
    import tunix.models.gemma4.model as M
    import tunix.models.gemma4.params_safetensors, tunix.sft.utils
    qwix.LoraProvider(module_path=".*", rank=2, alpha=4.0)
    g = jax.jit(jax.grad(lambda w: (w ** 2).sum()))(jnp.ones(4))
    tx = optax.adam(1e-3); st = tx.init(g); tx.update(g, st)
    cfg = M.ModelConfig.gemma4_e4b()
    out.update(ok=bool(float(g.sum()) == 8.0), jax=jax.__version__, flax=flax.__version__, optax=optax.__version__,
               e4b_layers=cfg.num_layers, libtpu=importlib.util.find_spec("libtpu") is not None)
except Exception:
    out["error"] = traceback.format_exc()[-1800:]
print("SMOKE_JSON:" + json.dumps(out))
sys.exit(0 if out["ok"] else 1)
"""


def smoke_test_tpu(fakes: dict) -> dict:
    """Install the TPU wheelhouse into a clean venv (target-owned numpy/safetensors from PyPI, as on the image) and import
    the exact modules the Tunix trainer uses. JAX runs on CPU here; libtpu only has to be present."""
    py = make_venv(SCRATCH / "smoke_tpu_venv")
    pins = [f"numpy=={fakes['numpy']}" if fakes.get("numpy") else "numpy",
            f"safetensors=={fakes.get('safetensors', OFFICIAL_FALLBACK['safetensors'])}"]
    pip(py, "install", "--quiet", *pins)
    pip(py, "install", "--quiet", "--no-index", "--no-deps", *map(str, sorted(OUT_TPU.glob("*.whl"))))
    chk = pip(py, "check", capture=True, check=False)
    issues = [l for l in chk.stdout.splitlines() if l.strip() and not l.startswith("No broken")]
    r = run([py, "-I", "-c", TPU_SMOKE], capture=True, check=False)
    log(tail(r.stderr, 6))
    line = next((l for l in r.stdout.splitlines() if l.startswith("SMOKE_JSON:")), None)
    if not line:
        raise StepError("TPU smoke test produced no result:\n" + tail(r.stdout + r.stderr))
    result = json.loads(line[len("SMOKE_JSON:"):])
    result["pip_check_lines"] = len(issues)
    if not result.get("ok"):
        raise StepError("TPU smoke test failed:\n" + json.dumps(result, indent=1)[:2500])
    if not result.get("libtpu"):
        raise StepError("libtpu is missing from the TPU wheelhouse: jax[tpu] would fall back to CPU on the TPU VM")
    return result


def build_tpu_bundle(fakes: dict, official_versions: dict, resolve_py: Path) -> dict:
    """JAX + Tunix wheelhouse in OUT_TPU. google-tunix is built --no-deps (see TUNIX_PIN); everything it imports is
    resolved explicitly and verified by `smoke_test_tpu`."""
    inherited = preinstall_requirements(official_versions, fakes)
    attempts, chosen = [], None
    for label, base_reqs in TPU_CANDIDATES:
        reqs = base_reqs + inherited
        log(f"\n=== TPU candidate '{label}': {reqs}")
        try:
            bundle = resolve(reqs, resolve_py, fakes)
            log(f"{len(bundle)} packages to bundle (+ {TUNIX_PIN} --no-deps)")
            build_wheels(bundle, OUT_TPU, extra_specs=(TUNIX_PIN,))
            closure = closure_check(fakes, OUT_TPU, frozenset(norm(n) for n in TPU_WAIVED))
            if closure["missing"] or closure["conflicts"]:
                raise StepError("TPU closure check failed: " + json.dumps(closure, indent=1))
            smoke = smoke_test_tpu(fakes)
            chosen = {"label": label, "requirements": base_reqs, "bundle": bundle, "smoke": smoke, "waived": closure["waived"]}
            attempts.append({"label": label, "ok": True})
            break
        except (StepError, subprocess.CalledProcessError) as exc:
            log(f"TPU candidate '{label}' REJECTED: {exc}")
            attempts.append({"label": label, "ok": False, "error": str(exc)[-2000:]})
    if not chosen:
        raise StepError("No TPU candidate passed verification. See the log above.")
    wheels = sorted(OUT_TPU.glob("*.whl"))
    (OUT_TPU / "MANIFEST.json").write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
        "chosen": chosen["label"], "requirements": chosen["requirements"], "tunix": TUNIX_PIN,
        "waived_dependencies": chosen["waived"], "wheels": {w.name: sha256(w) for w in wheels},
        "smoke": chosen["smoke"], "attempts": attempts,
    }, indent=2))
    (OUT_TPU / "requirements.lock.txt").write_text("".join(f"{n}=={v}\n" for n, v in chosen["bundle"]) + TUNIX_PIN + "\n")
    log(f"\nTPU VERIFIED ({chosen['label']}): {len(wheels)} wheels, {sum(w.stat().st_size for w in wheels) / 1e6:.0f} MB in {OUT_TPU}")
    for k in ("jax", "flax", "optax", "e4b_layers", "libtpu"):
        log(f"  {k:12s} {chosen['smoke'].get(k)}")
    return chosen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    global PY
    log(f"notebook python {platform.python_version()} on {platform.platform()}")
    PY = ensure_py312()
    py_ok = True
    official_dirs, official_versions = find_official_wheels()
    log(f"official wheelhouse attached: {bool(official_dirs)} ({len(official_versions)} wheels)")
    fakes = {**OFFICIAL_FALLBACK, **official_versions}
    for name, version in installed_versions(BASE_IMAGE).items():
        fakes[name] = version
    log(f"target-owned packages modelled: {len(fakes)} (torch={fakes.get('torch')}, numpy={fakes.get('numpy')})")
    resolve_py = make_resolve_venv(fakes)

    failures = []
    if "tpu" in BUNDLES:
        try:
            build_tpu_bundle(fakes, official_versions, resolve_py)
        except (StepError, subprocess.CalledProcessError) as exc:
            failures.append(f"tpu bundle: {exc}")
            log(f"TPU BUNDLE FAILED (the core bundle is still built): {exc}")
    if "core" not in BUNDLES:
        if failures:
            raise SystemExit("\n".join(failures))
        log("\nNext: Save Version (Save & Run All) -> Output -> New Dataset version -> send me the dataset name.")
        return

    inherited = preinstall_requirements(official_versions, fakes)
    attempts, chosen = [], None
    for label, base_reqs in CANDIDATES:
        reqs = base_reqs + inherited
        log(f"\n=== candidate '{label}': {reqs}")
        try:
            bundle = resolve(reqs, resolve_py, fakes)
            log(f"{len(bundle)} packages to bundle: {[f'{n}=={v}' for n, v in bundle]}")
            build_wheels(bundle)
            closure = closure_check(fakes)
            if closure["missing"] or closure["conflicts"]:
                raise StepError("closure check failed: " + json.dumps(closure, indent=1))
            smoke = smoke_test(official_dirs, official_versions, {n for n, _ in bundle}, fakes)
            chosen = {"label": label, "requirements": base_reqs, "inherited": inherited, "bundle": bundle, "smoke": smoke}
            attempts.append({"label": label, "ok": True})
            break
        except (StepError, subprocess.CalledProcessError) as exc:
            log(f"candidate '{label}' REJECTED: {exc}")
            attempts.append({"label": label, "ok": False, "error": str(exc)[-2000:]})
    if not chosen:
        raise SystemExit("No candidate passed verification. See the log above; send it to me.")

    wheels = sorted(OUT.glob("*.whl"))
    manifest = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "python_matches_target": py_ok,
        "platform": platform.platform(),
        "official_wheelhouse_attached": bool(official_dirs),
        "target_owned_versions": {k: fakes[k] for k in sorted(fakes) if k in BASE_IMAGE or k in PREINSTALL},
        "chosen": chosen["label"],
        "requirements": chosen["requirements"],
        "wheels": {w.name: sha256(w) for w in wheels},
        "smoke": chosen["smoke"],
        "attempts": attempts,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "requirements.lock.txt").write_text("".join(f"{n}=={v}\n" for n, v in chosen["bundle"]))
    size_mb = sum(w.stat().st_size for w in wheels) / 1e6
    log(f"\nVERIFIED ({chosen['label']}): {len(wheels)} wheels, {size_mb:.1f} MB in {OUT}")
    for key in ("peft", "trl", "accelerate", "datasets", "optuna", "wandb", "transformers"):
        log(f"  {key:13s} {chosen['smoke']['core'].get(key)}")
    log(f"  tiny SFT end-to-end: {chosen['smoke']['e2e']}")
    if failures:
        log("\nWARNING: " + "; ".join(failures))
    log("Next: Save Version (Save & Run All) -> Output -> New Dataset (version) containing train_wheels/ and train_wheels_tpu/.")


main()
