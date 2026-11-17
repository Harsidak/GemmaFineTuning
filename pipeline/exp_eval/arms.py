"""Experiment arms as overlays.

An arm lives in `experiments/arms/<arm_id>/arm.yaml`. It is either a full arm (`base: null`, its submission
directory sits next to the spec) or an overlay on another arm (`base: <arm_id>`) that declares every change:

    id: exp016_a1_skill_first
    base: exp016_a0_pruned
    changes:
      files:                         # destination in the submission -> source (repo-relative file or directory)
        prompts/system.md: pipeline/submission_template/prompts/system.md
        skills/swe-kit: data/Skills/swe-kit
      yaml_set:                      # submission YAML file -> {dotted.key: value}; a null value deletes the key
        agent.yaml: {skills: [skills/swe-kit]}
        eval_config.yaml: {evaluation.max_tool_calls: 30}
      remove: []                     # submission paths (files or directories) to delete

Building only ever applies the declared changes, and the build manifest lists exactly which files differ from the
base arm, so "one variable changed" can be read off the manifest instead of trusted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARMS_ROOT = REPO_ROOT / "experiments" / "arms"

# HARNESS_README.md §2.4 / CLAUDE.md: the only bases the scorer accepts, and the file types a submission may hold.
ALLOWED_MODELS = frozenset(
    {
        "gemma-4-31b-it-qat-w4a16-ct",
        "gemma-4-12b-it-qat-w4a16-ct",
        "gemma-4-e4b-it-qat-w4a16-ct",
        "gemma-4-e2b-it-qat-w4a16-ct",
    }
)
ALLOWED_EXTENSIONS = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})
OFFICIAL_TOOLS = frozenset(
    {
        "run_command",
        "submit_patch",
        "get_status",
        "read_file",
        "edit_file",
        "write_file",
        "search_similar_code",
        "get_code_neighbors",
        "get_code_subgraph",
    }
)
EVAL_CONFIG_KEYS = frozenset({"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"})
MAX_GENERATION_TOKENS = 32768
MAX_UNPACKED_BYTES = 3 * 1024**3
# Never copied into a submission: byte-code caches and editor / OS litter.
_IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", "Thumbs.db", ".ipynb_checkpoints"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class ArmError(ValueError):
    """An arm spec is malformed, or a declared change cannot be applied."""


# --------------------------------------------------------------------------------------------- YAML with !include


@dataclass(frozen=True)
class Include:
    """An ADK `!include <path>` tag, kept as a tag so a rewritten agent.yaml still includes the same file."""

    path: str


class _IncludeLoader(yaml.SafeLoader):
    pass


class _IncludeDumper(yaml.SafeDumper):
    pass


_IncludeLoader.add_constructor("!include", lambda loader, node: Include(loader.construct_scalar(node)))
_IncludeDumper.add_representer(Include, lambda dumper, inc: dumper.represent_scalar("!include", inc.path))


def load_yaml(path: Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_IncludeLoader)


_QUOTED_INCLUDE_RE = re.compile(r"!include '([A-Za-z0-9_./-]+)'")


def dump_yaml(data: Any) -> str:
    text = yaml.dump(data, Dumper=_IncludeDumper, sort_keys=False, default_flow_style=False, allow_unicode=True)
    # PyYAML always quotes a tagged scalar; write `!include prompts/system.md` exactly like the official sample.
    return _QUOTED_INCLUDE_RE.sub(r"!include \1", text)


def set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Sets `a.b.c` in nested dicts, creating missing levels; `None` deletes the key."""
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        nxt = node.get(k)
        if nxt is None:
            if value is None:
                return
            nxt = node[k] = {}
        if not isinstance(nxt, dict):
            raise ArmError(f"cannot set {dotted!r}: {k!r} is a {type(nxt).__name__}, not a mapping")
        node = nxt
    if value is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value


# ------------------------------------------------------------------------------------------------------ arm specs


@dataclass
class ArmSpec:
    arm_id: str
    spec_dir: Path
    base: str | None
    description: str = ""
    experiment: str = ""
    files: dict[str, str] = field(default_factory=dict)
    yaml_set: dict[str, dict[str, Any]] = field(default_factory=dict)
    remove: list[str] = field(default_factory=list)

    @property
    def declared_paths(self) -> list[str]:
        return sorted({*self.files, *self.yaml_set, *self.remove})


def load_arm_spec(arm_id: str, arms_root: Path = DEFAULT_ARMS_ROOT) -> ArmSpec:
    spec_path = Path(arms_root) / arm_id / "arm.yaml"
    if not spec_path.is_file():
        known = sorted(p.parent.name for p in Path(arms_root).glob("*/arm.yaml"))
        raise ArmError(f"no arm {arm_id!r} ({spec_path} missing); known arms: {known}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if raw.get("id", arm_id) != arm_id:
        raise ArmError(f"{spec_path}: id {raw.get('id')!r} does not match its folder name {arm_id!r}")
    changes = raw.get("changes") or {}
    unknown = set(changes) - {"files", "yaml_set", "remove"}
    if unknown:
        raise ArmError(f"{spec_path}: unknown change kinds {sorted(unknown)}")
    spec = ArmSpec(
        arm_id=arm_id,
        spec_dir=spec_path.parent,
        base=raw.get("base"),
        description=str(raw.get("description", "")).strip(),
        experiment=str(raw.get("experiment", "")),
        files={_clean_rel(k): str(v) for k, v in (changes.get("files") or {}).items()},
        yaml_set={_clean_rel(k): dict(v or {}) for k, v in (changes.get("yaml_set") or {}).items()},
        remove=[_clean_rel(p) for p in (changes.get("remove") or [])],
    )
    if spec.base is None:
        if spec.files or spec.yaml_set or spec.remove:
            raise ArmError(f"{spec_path}: a full arm (base: null) cannot declare changes")
        if not (spec.spec_dir / "submission" / "agent.yaml").is_file():
            raise ArmError(f"{spec_path}: a full arm needs {spec.spec_dir / 'submission' / 'agent.yaml'}")
    elif not spec.declared_paths:
        raise ArmError(f"{spec_path}: an overlay arm must declare at least one change")
    return spec


def _clean_rel(p: str) -> str:
    rel = str(p).replace("\\", "/").strip().strip("/")
    if not rel or rel.startswith("..") or "/../" in f"/{rel}/" or Path(rel).is_absolute():
        raise ArmError(f"invalid submission path {p!r}")
    return rel


def arm_chain(arm_id: str, arms_root: Path = DEFAULT_ARMS_ROOT) -> list[ArmSpec]:
    """[full root arm, ..., arm_id] in build order."""
    chain: list[ArmSpec] = []
    seen: set[str] = set()
    cur: str | None = arm_id
    while cur is not None:
        if cur in seen:
            raise ArmError(f"base cycle through {cur!r}")
        seen.add(cur)
        spec = load_arm_spec(cur, arms_root)
        chain.append(spec)
        cur = spec.base
    return chain[::-1]


# ------------------------------------------------------------------------------------------------------ building


def _ignored(p: Path) -> bool:
    return p.name in _IGNORED_NAMES or p.suffix in _IGNORED_SUFFIXES


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if _ignored(Path(d) / n)])


def file_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not any(part in _IGNORED_NAMES for part in rel.parts) and p.suffix not in _IGNORED_SUFFIXES:
            out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _resolve_source(src: str, repo_root: Path) -> Path:
    p = (repo_root / src).resolve()
    if not p.exists():
        raise ArmError(f"change source {src!r} does not exist under {repo_root}")
    return p


def _apply_layer(spec: ArmSpec, sub: Path, repo_root: Path) -> list[str]:
    """Applies one overlay to the submission dir `sub`; returns declared changes that had no effect."""
    before = file_hashes(sub)
    for rel in spec.remove:
        target = sub / rel
        if not target.exists():
            raise ArmError(f"{spec.arm_id}: remove {rel!r} but it is not in the base submission")
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    for rel, src in spec.files.items():
        _copy_tree(_resolve_source(src, repo_root), sub / rel)
    for rel, sets in spec.yaml_set.items():
        target = sub / rel
        data = load_yaml(target) if target.is_file() else {}
        if not isinstance(data, dict):
            raise ArmError(f"{spec.arm_id}: {rel} is not a YAML mapping")
        original = copy.deepcopy(data)
        for key, value in sets.items():
            set_dotted(data, str(key), value)
        if data == original and target.is_file():
            continue  # same values: keep the base file byte for byte (reported below as a no-op change)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dump_yaml(data), encoding="utf-8")
    after = file_hashes(sub)
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    return [rel for rel in spec.declared_paths if not any(c == rel or c.startswith(rel + "/") for c in changed)]


@dataclass
class BuiltArm:
    arm_id: str
    submission_dir: Path
    zip_path: Path
    manifest: dict[str, Any]


def build_arm(
    arm_id: str,
    out_dir: Path,
    *,
    arms_root: Path = DEFAULT_ARMS_ROOT,
    repo_root: Path = REPO_ROOT,
) -> BuiltArm:
    """Builds `out_dir/submission/` + `out_dir/submission.zip` + `out_dir/arm_manifest.json` for `arm_id`.

    Raises ArmError when a change cannot be applied or the result breaks a competition rule."""
    out_dir = Path(out_dir)
    sub = out_dir / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    chain = arm_chain(arm_id, arms_root)
    _copy_tree(chain[0].spec_dir / "submission", sub)
    base_hashes = file_hashes(sub)
    no_effect: list[str] = []
    for spec in chain[1:]:
        base_hashes = file_hashes(sub)
        no_effect = _apply_layer(spec, sub, Path(repo_root))
    final = file_hashes(sub)
    target = chain[-1]
    if target.base is not None and no_effect:
        raise ArmError(f"{arm_id}: declared changes {no_effect} leave the base arm unchanged; drop them")

    errors, warnings = check_submission_rules(sub)
    if errors:
        raise ArmError(f"{arm_id}: submission breaks competition rules: {errors}")
    zip_path = zip_submission(sub, out_dir / "submission.zip")
    manifest = {
        "arm_id": arm_id,
        "experiment": target.experiment,
        "description": target.description,
        "chain": [s.arm_id for s in chain],
        "base": target.base,
        "declared_changes": target.declared_paths,
        "changed_vs_base": sorted(k for k in base_hashes.keys() | final.keys() if base_hashes.get(k) != final.get(k))
        if target.base is not None
        else [],
        "files": final,
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "zip_bytes": zip_path.stat().st_size,
        "eval_config": read_eval_config(sub),
        "rule_warnings": warnings,
    }
    (out_dir / "arm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return BuiltArm(arm_id=arm_id, submission_dir=sub, zip_path=zip_path, manifest=manifest)


def zip_submission(sub: Path, zip_path: Path) -> Path:
    """Deterministic zip (sorted entries, fixed timestamps), so the same arm always has the same sha256."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(file_hashes(sub)):
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (sub / rel).read_bytes())
    return zip_path


# ------------------------------------------------------------------------------------------------- rule checking

# Harness defaults (HARNESS_README §7.1, scripts/inference.py) for keys an eval_config.yaml leaves out.
HARNESS_EVAL_DEFAULTS = {"timeout_seconds": 300, "max_tool_calls": 100, "max_time_minutes": 60.0, "max_turns": 500}


def read_eval_config(sub: Path) -> dict[str, Any]:
    """The per-task budget the scorer will apply: the arm's eval_config.yaml over the harness defaults."""
    out = dict(HARNESS_EVAL_DEFAULTS)
    path = Path(sub) / "eval_config.yaml"
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        section = raw.get("evaluation", raw) if isinstance(raw, dict) else {}
        out.update({k: v for k, v in section.items() if k in EVAL_CONFIG_KEYS and v is not None})
    return out


_PROMPT_CALLS_RE = re.compile(r"\b(\d{1,3})\s+tool\s+calls\b", re.IGNORECASE)
_PROMPT_MINUTES_RE = re.compile(r"\b(\d{1,3})\s+minutes\b", re.IGNORECASE)


def check_submission_rules(sub: Path) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a submission directory. Errors are things the scorer rejects or that silently break a
    run; warnings are inconsistencies worth a look (e.g. the prompt promising a different budget than eval_config)."""
    sub = Path(sub)
    errors: list[str] = []
    warnings: list[str] = []
    files = file_hashes(sub)
    total = sum((sub / f).stat().st_size for f in files)
    if total >= MAX_UNPACKED_BYTES:
        errors.append(f"unpacked size {total} B is not below 3 GiB")
    for rel in files:
        suffix = Path(rel).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            errors.append(f"{rel}: extension {suffix or '(none)'} is not allowed")
        if suffix == ".py" and not rel.startswith("skills/"):
            errors.append(f"{rel}: .py files are only allowed inside skills/")

    agent_path = sub / "agent.yaml"
    if not agent_path.is_file():
        return [*errors, "missing root agent.yaml"], warnings
    agent = load_yaml(agent_path)
    if not isinstance(agent, dict):
        return [*errors, "agent.yaml is not a mapping"], warnings

    models = {agent.get("model")}
    for sa in sorted((sub / "sub_agents").glob("*.yaml")) if (sub / "sub_agents").is_dir() else []:
        sd = load_yaml(sa)
        if isinstance(sd, dict) and sd.get("model"):
            models.add(sd["model"])
    if len(models) != 1 or not models <= ALLOWED_MODELS:
        errors.append(f"agents must declare one allowed base model, found {sorted(map(str, models))}")

    tools = agent.get("tools") or []
    bad_tools = [t for t in tools if not isinstance(t, str) or t not in OFFICIAL_TOOLS]
    if bad_tools:
        errors.append(f"agent.yaml declares tools the harness does not have: {bad_tools}")

    for key, value in agent.items():
        if isinstance(value, Include) and not (sub / value.path).is_file():
            errors.append(f"agent.yaml {key}: !include {value.path} does not exist")

    adapter = agent.get("adapter")
    if adapter and not (sub / "adapters" / str(adapter)).is_dir():
        errors.append(f"agent.yaml adapter {adapter!r} has no adapters/{adapter}/ directory")

    for skill in agent.get("skills") or []:
        if not (sub / str(skill) / "SKILL.md").is_file():
            errors.append(f"skill {skill!r} has no SKILL.md")
    declared_skills = {str(s).strip("/") for s in agent.get("skills") or []}
    for skill_md in sub.glob("skills/*/SKILL.md"):
        rel = skill_md.parent.relative_to(sub).as_posix()
        if rel not in declared_skills:
            warnings.append(f"{rel} is shipped but not listed under agent.yaml skills:")

    gen = agent.get("generate_content_config")
    gen_cfg = load_yaml(sub / gen.path) if isinstance(gen, Include) and (sub / gen.path).is_file() else gen
    if isinstance(gen_cfg, dict):
        mot = gen_cfg.get("max_output_tokens")
        tb = (gen_cfg.get("thinking_config") or {}).get("thinking_budget")
        for name, val in (("max_output_tokens", mot), ("thinking_budget", tb)):
            if val is not None and not (0 <= int(val) <= MAX_GENERATION_TOKENS):
                errors.append(f"{name}={val} outside 0..{MAX_GENERATION_TOKENS}")

    ec_path = sub / "eval_config.yaml"
    if ec_path.is_file():
        raw = yaml.safe_load(ec_path.read_text(encoding="utf-8")) or {}
        section = raw.get("evaluation") if isinstance(raw, dict) else None
        if not isinstance(section, dict):
            errors.append("eval_config.yaml has no `evaluation:` mapping")
        else:
            unknown = set(section) - EVAL_CONFIG_KEYS
            if unknown:
                errors.append(f"eval_config.yaml has keys the harness ignores: {sorted(unknown)}")

    instr = agent.get("instruction")
    if isinstance(instr, Include) and (sub / instr.path).is_file():
        prompt = (sub / instr.path).read_text(encoding="utf-8", errors="replace")
        budget = read_eval_config(sub)
        said_calls = {int(m) for m in _PROMPT_CALLS_RE.findall(prompt)}
        if said_calls and int(budget["max_tool_calls"]) not in said_calls:
            warnings.append(f"prompt mentions {sorted(said_calls)} tool calls, eval_config max_tool_calls="
                            f"{budget['max_tool_calls']}")
        said_min = {int(m) for m in _PROMPT_MINUTES_RE.findall(prompt)}
        if said_min and round(float(budget["max_time_minutes"])) not in said_min:
            warnings.append(f"prompt mentions {sorted(said_min)} minutes, eval_config max_time_minutes="
                            f"{budget['max_time_minutes']}")
        mentioned = {t for t in OFFICIAL_TOOLS if f"`{t}`" in prompt or f" {t} " in prompt}
        undeclared = sorted(mentioned - set(tools))
        if undeclared:
            warnings.append(f"prompt mentions tools the agent does not declare: {undeclared}")
    return errors, warnings


def list_arms(arms_root: Path = DEFAULT_ARMS_ROOT) -> list[str]:
    return sorted(p.parent.name for p in Path(arms_root).glob("*/arm.yaml"))
