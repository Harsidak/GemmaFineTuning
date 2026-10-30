"""Phase 5 (`package`): stage submission template, resolve adapters with 4-tier fallback, validate, zip.

Implements design_doc.md §6.2 and §8.2:
- Guarantees Goal G4 ("Always ship"): even if training or export failed completely, resolves each
  adapter slot via:
    (1) `runs/<run_id>/adapters/<target>`
    (2) newest checkpoint converted via `export_single_adapter`
    (3) `submission.fallback_adapters_dir`
    (4) bare base model (removes `adapter:` line from YAML as text, preserving `!include`)
- Validates with `pipeline.src.utils.packager.validate_submission_directory`,
  `adk_submission.validate_directory` (when installed), and strict competition rules.
- Writes `runs/<run_id>/submission.zip` and copies to `submission.working_zip_path` (`/kaggle/working/submission.zip`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from ...src.agent.builder import ADKSubmissionBuilder
from ...src.utils.packager import (
    ADKYamlLoader,
    package_submission_archive,
    validate_submission_directory,
)
from ..config import (
    ADAPTER_NAME_RE,
    ALLOWED_SERVED_MODELS,
    MAX_CONTEXT_TOKENS,
    MAX_LORA_RANK,
    MAX_SUBMISSION_BYTES,
    PipelineConfig,
    load_config,
)
from ..runctx import (
    PhaseStatus,
    RunContext,
    configure_phase_logging,
    sha256_file,
    utc_iso,
)
from .export import export_single_adapter, locate_trained_adapter_source

ALLOWED_SUBMISSION_EXTENSIONS: frozenset[str] = frozenset(
    {".yaml", ".yml", ".md", ".txt", ".json", ".safetensors"}
)
INCLUDE_TAG_RE = re.compile(r"!include\s+([^\s#]+)")
MODEL_LINE_RE = re.compile(r"^(\s*model\s*:\s*)([^\n#]+?)(\s*(?:#.*)?)$", re.MULTILINE)
ADAPTER_LINE_RE = re.compile(r"^(\s*adapter\s*:\s*)([^\n#]+?)(\s*(?:#.*)?)$", re.MULTILINE)


class SubmissionValidationError(ValueError):
    """Raised when a staged submission violates competition structural, model, or size rules."""


def default_sample_template_dir() -> Path | None:
    """Returns `build/gemma-4-developer-agent/sample_submission` when running inside the repo."""
    repo_root = Path(__file__).resolve().parents[3]
    cand = repo_root / "build" / "gemma-4-developer-agent" / "sample_submission"
    return cand if cand.is_dir() else None


def copy_template_without_adapters(template_dir: Path, dst_dir: Path) -> None:
    """Copies `template_dir` into `dst_dir` while omitting `adapters/` and rejecting symlinks."""
    shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_item in sorted(template_dir.rglob("*")):
        rel = src_item.relative_to(template_dir)
        if rel.parts and rel.parts[0] == "adapters":
            continue
        if src_item.is_symlink():
            raise SubmissionValidationError(f"Template contains forbidden symlink: {src_item}")
        target = dst_dir / rel
        if src_item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_item, target)


def rewrite_yaml_model_line(yaml_path: Path, served_model_id: str) -> bool:
    """Rewrites `model: ...` in `yaml_path` as raw text so `!include` tags remain untouched (§8.2)."""
    text = yaml_path.read_text(encoding="utf-8")

    def _repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{served_model_id}{m.group(3)}"

    new_text, count = MODEL_LINE_RE.subn(_repl, text)
    if count > 0 and new_text != text:
        yaml_path.write_text(new_text, encoding="utf-8")
    return count > 0


def find_referenced_adapters_in_yaml(yaml_path: Path) -> list[str]:
    """Extracts `adapter: <name>` values declared in an agent YAML file."""
    text = yaml_path.read_text(encoding="utf-8")
    found: list[str] = []
    for m in ADAPTER_LINE_RE.finditer(text):
        val = m.group(2).strip().strip("'\"")
        if val and val.lower() not in {"null", "none", "~"}:
            found.append(val)
    return found


def remove_yaml_adapter_line(yaml_path: Path, slot_name: str) -> bool:
    """Removes `adapter: <slot_name>` line as text for bare-base-model fallback, preserving `!include`."""
    lines = yaml_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    removed = False
    for line in lines:
        m = ADAPTER_LINE_RE.match(line.rstrip("\r\n"))
        if m:
            val = m.group(2).strip().strip("'\"")
            if val == slot_name:
                removed = True
                continue
        kept.append(line)
    if removed:
        yaml_path.write_text("".join(kept), encoding="utf-8")
    return removed


def _agent_yaml_files(submission_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"):
        p = submission_dir / root_name
        if p.is_file():
            files.append(p)
    sub_dir = submission_dir / "sub_agents"
    if sub_dir.is_dir():
        for ext in ("*.yaml", "*.yml"):
            files.extend(sorted(sub_dir.rglob(ext)))
    return files


def _check_generation_ceilings(node: Any, source_label: str) -> None:
    """Recursively checks `max_output_tokens <= 32768` and `thinking_budget <= 32768` (§8.2)."""
    if isinstance(node, dict):
        if "max_output_tokens" in node and node["max_output_tokens"] is not None:
            mot = int(node["max_output_tokens"])
            if not (1 <= mot <= MAX_CONTEXT_TOKENS):
                raise SubmissionValidationError(
                    f"{source_label}: max_output_tokens={mot} outside 1..{MAX_CONTEXT_TOKENS}"
                )
        if "thinking_budget" in node and node["thinking_budget"] is not None:
            tb = int(node["thinking_budget"])
            if not (0 <= tb <= MAX_CONTEXT_TOKENS):
                raise SubmissionValidationError(
                    f"{source_label}: thinking_budget={tb} outside 0..{MAX_CONTEXT_TOKENS}"
                )
        for k, v in node.items():
            _check_generation_ceilings(v, f"{source_label}.{k}")
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _check_generation_ceilings(item, f"{source_label}[{idx}]")


def validate_staged_submission(submission_dir: Path, expected_served_model: str | None = None) -> dict[str, Any]:
    """Runs all 3 required validators on `submission_dir` (§8.2)."""
    # 1. Existing shared packager validation (`pipeline.src.utils.packager`)
    is_valid, pkg_errors = validate_submission_directory(submission_dir)
    if not is_valid:
        raise SubmissionValidationError(
            f"Shared packager validation failed: {'; '.join(pkg_errors)}"
        )

    # 2. Official `adk_submission.validate_directory` when installed in the environment
    official_validated = False
    try:
        import adk_submission  # type: ignore[import-not-found]

        if hasattr(adk_submission, "validate_directory"):
            adk_submission.validate_directory(submission_dir)
            official_validated = True
    except ImportError:
        pass

    # 3. Strict competition rules: root config count, symlinks, extensions, !include traversal, size < 3 GiB, tokens, adapters
    root_configs = [
        p
        for p in (
            submission_dir / "agent.yaml",
            submission_dir / "agent.yml",
            submission_dir / "root_agent.yaml",
            submission_dir / "root_agent.yml",
        )
        if p.exists()
    ]
    if len(root_configs) != 1:
        raise SubmissionValidationError(
            f"Expected exactly 1 root agent config in {submission_dir}, found {len(root_configs)}"
        )

    total_bytes = 0
    file_count = 0
    for item in sorted(submission_dir.rglob("*")):
        if item.is_symlink():
            raise SubmissionValidationError(f"Symlink forbidden in submission: {item}")
        if not item.is_file():
            continue
        file_count += 1
        total_bytes += item.stat().st_size
        rel = item.relative_to(submission_dir)
        ext = item.suffix.lower()
        if ext == ".py":
            if not (len(rel.parts) >= 2 and rel.parts[0] == "skills"):
                raise SubmissionValidationError(
                    f"Python file forbidden outside skills/: {rel.as_posix()}"
                )
        elif ext not in ALLOWED_SUBMISSION_EXTENSIONS:
            raise SubmissionValidationError(
                f"Disallowed file extension {ext!r} in submission: {rel.as_posix()}"
            )

        if ext in {".yaml", ".yml"}:
            raw_txt = item.read_text(encoding="utf-8")
            for inc_match in INCLUDE_TAG_RE.finditer(raw_txt):
                inc_target = inc_match.group(1).strip("'\"")
                if (
                    "\x00" in inc_target
                    or inc_target.startswith(("/", "\\"))
                    or Path(inc_target).is_absolute()
                ):
                    raise SubmissionValidationError(
                        f"Unsafe !include path {inc_target!r} in {rel.as_posix()}"
                    )
                inc_resolved = (item.parent / inc_target).resolve()
                if not inc_resolved.is_relative_to(submission_dir.resolve()):
                    raise SubmissionValidationError(
                        f"!include escapes submission root in {rel.as_posix()}: {inc_target!r}"
                    )
                if not inc_resolved.is_file():
                    raise SubmissionValidationError(
                        f"!include target missing in {rel.as_posix()}: {inc_target!r}"
                    )
            parsed = yaml.load(raw_txt, Loader=ADKYamlLoader)
            _check_generation_ceilings(parsed, rel.as_posix())

    if total_bytes >= MAX_SUBMISSION_BYTES:
        raise SubmissionValidationError(
            f"Total unpacked submission size {total_bytes} bytes exceeds 3 GiB limit ({MAX_SUBMISSION_BYTES})"
        )

    # Validate single declared model & adapter references across all agent YAMLs
    declared_models: set[str] = set()
    referenced_adapters: set[str] = set()
    for afile in _agent_yaml_files(submission_dir):
        parsed = yaml.load(afile.read_text(encoding="utf-8"), Loader=ADKYamlLoader)
        if not isinstance(parsed, dict):
            continue
        m_val = parsed.get("model")
        if isinstance(m_val, str) and m_val.strip():
            norm_m = re.sub(r"^(openai|google|hosted_vllm|custom)/", "", m_val.strip())
            declared_models.add(norm_m)
        a_val = parsed.get("adapter")
        if isinstance(a_val, str) and a_val.strip():
            referenced_adapters.add(a_val.strip())

    if len(declared_models) != 1:
        raise SubmissionValidationError(
            f"Submission must declare exactly 1 base model across agents, got {sorted(declared_models)}"
        )
    sole_model = next(iter(declared_models))
    if sole_model not in ALLOWED_SERVED_MODELS:
        raise SubmissionValidationError(
            f"Declared model {sole_model!r} is not in allowed competition models {sorted(ALLOWED_SERVED_MODELS)}"
        )
    if expected_served_model and sole_model != expected_served_model:
        raise SubmissionValidationError(
            f"Declared model {sole_model!r} != configured served model {expected_served_model!r}"
        )

    adapters_root = submission_dir / "adapters"
    present_adapters: set[str] = set()
    if adapters_root.is_dir():
        for sub in sorted(adapters_root.iterdir()):
            if not sub.is_dir():
                raise SubmissionValidationError(f"Unexpected file directly under adapters/: {sub.name}")
            if not ADAPTER_NAME_RE.match(sub.name):
                raise SubmissionValidationError(f"Invalid adapter directory name: {sub.name!r}")
            acfg_path = sub / "adapter_config.json"
            ast_path = sub / "adapter_model.safetensors"
            if not acfg_path.is_file() or not ast_path.is_file():
                raise SubmissionValidationError(
                    f"Adapter {sub.name!r} must contain adapter_config.json and adapter_model.safetensors"
                )
            acfg = json.loads(acfg_path.read_text(encoding="utf-8"))
            r_val = int(acfg.get("r", 0))
            if not (1 <= r_val <= MAX_LORA_RANK):
                raise SubmissionValidationError(
                    f"Adapter {sub.name!r} rank r={r_val} outside 1..{MAX_LORA_RANK}"
                )
            present_adapters.add(sub.name)

    missing_refs = sorted(referenced_adapters - present_adapters)
    if missing_refs:
        raise SubmissionValidationError(
            f"Agent YAML references adapter(s) missing from adapters/: {missing_refs}"
        )

    return {
        "official_validator_ran": official_validated,
        "declared_model": sole_model,
        "present_adapters": sorted(present_adapters),
        "referenced_adapters": sorted(referenced_adapters),
        "file_count": file_count,
        "unpacked_size_bytes": total_bytes,
    }


def _resolve_slot_adapter(
    *,
    ctx: RunContext,
    cfg: PipelineConfig,
    slot_name: str,
    target_name: str,
    served_id: str,
) -> tuple[Path | None, str]:
    """Resolves an adapter directory for `slot_name -> target_name` following §6.2 fallback priority."""
    # Tier 1: Exported adapter in `runs/<run_id>/adapters/<target_name>`
    exp_dir = ctx.adapters_dir / target_name
    if (exp_dir / "adapter_model.safetensors").is_file() and (exp_dir / "adapter_config.json").is_file():
        return exp_dir, f"exported:{target_name}"

    # Tier 2: Latest checkpoint converted on the fly via `export_single_adapter`
    ckpt_src, ckpt_kind = locate_trained_adapter_source(ctx, target_name)
    if ckpt_src is not None:
        try:
            eff = cfg.resolve_adapter(target_name) if any(a.name == target_name for a in cfg.adapters) else None
            recovered_dir = ctx.adapters_dir / f"_recovered_{target_name}"
            export_single_adapter(
                adapter_name=target_name,
                source_adapter_dir=ckpt_src,
                output_adapter_dir=recovered_dir,
                served_model_id=served_id,
                eff_adapter=eff,
            )
            return recovered_dir, f"checkpoint_recovery:{target_name}({ckpt_kind})"
        except Exception:  # noqa: BLE001, S110
            pass

    # Tier 3: Fallback pre-built adapters directory (`submission.fallback_adapters_dir`)
    if cfg.submission.fallback_adapters_dir:
        fb_root = Path(cfg.submission.fallback_adapters_dir)
        for cand_name in (target_name, slot_name):
            cand = fb_root / cand_name
            if (cand / "adapter_model.safetensors").is_file() and (cand / "adapter_config.json").is_file():
                return cand, f"fallback_dir:{cand}"

    # Tier 4: Bare base model (strip `adapter:` reference from YAML)
    return None, "bare_base_model"


def run_package_phase(
    ctx: RunContext,
    cfg: PipelineConfig | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Stages `runs/<run_id>/submission/`, resolves adapters, validates, and builds `submission.zip`."""
    logger = configure_phase_logging(ctx.log_path("package"))
    t_start = time.time()
    cfg = cfg or load_config(ctx.resolved_config_path)
    served_id = cfg.effective_served_model_id()

    notes: list[str] = []
    adapter_sources: dict[str, str] = {}

    try:
        # 1. Stage template
        if cfg.submission.template_dir:
            tpl_dir = Path(cfg.submission.template_dir)
            if not tpl_dir.is_dir():
                raise FileNotFoundError(f"submission.template_dir not found: {tpl_dir}")
            copy_template_without_adapters(tpl_dir, ctx.submission_dir)
            template_source = str(tpl_dir)
        else:
            sample_tpl = default_sample_template_dir()
            if sample_tpl is not None:
                copy_template_without_adapters(sample_tpl, ctx.submission_dir)
                template_source = str(sample_tpl)
            else:
                shutil.rmtree(ctx.submission_dir, ignore_errors=True)
                builder = ADKSubmissionBuilder(base_model_name=served_id, output_dir=ctx.submission_dir)
                builder.build()
                template_source = "ADKSubmissionBuilder.build"

        # 2. Rewrite `model:` in root and sub-agent YAMLs preserving `!include`
        agent_files = _agent_yaml_files(ctx.submission_dir)
        referenced_slots: list[str] = []
        for afile in agent_files:
            rewrite_yaml_model_line(afile, served_id)
            for s in find_referenced_adapters_in_yaml(afile):
                if s not in referenced_slots:
                    referenced_slots.append(s)

        # Include any additional slots explicitly listed in `submission.adapter_map` that match YAML refs
        for slot_name in referenced_slots:
            target_name = cfg.submission.adapter_map.get(slot_name, slot_name)
            resolved_path, source_desc = _resolve_slot_adapter(
                ctx=ctx,
                cfg=cfg,
                slot_name=slot_name,
                target_name=target_name,
                served_id=served_id,
            )
            adapter_sources[slot_name] = source_desc

            if resolved_path is not None:
                dst_slot = ctx.submission_dir / "adapters" / slot_name
                shutil.rmtree(dst_slot, ignore_errors=True)
                dst_slot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(resolved_path, dst_slot)
                if not source_desc.startswith("exported:"):
                    msg = f"Slot {slot_name!r} populated via fallback {source_desc}"
                    notes.append(msg)
                    logger.warning(msg)
            else:
                for afile in agent_files:
                    remove_yaml_adapter_line(afile, slot_name)
                msg = f"Slot {slot_name!r} had no usable adapter; stripped 'adapter: {slot_name}' (bare base model)"
                notes.append(msg)
                logger.warning(msg)

        # Remove empty `adapters/` folder if all slots fell back to bare base model
        adapters_dir = ctx.submission_dir / "adapters"
        if adapters_dir.is_dir() and not any(adapters_dir.iterdir()):
            adapters_dir.rmdir()

        # 3. Validate staged submission
        validation_info = validate_staged_submission(ctx.submission_dir, expected_served_model=served_id)

        # 4. Create `runs/<run_id>/submission.zip` and copy to `/kaggle/working/submission.zip`
        package_submission_archive(ctx.submission_dir, ctx.submission_zip_path)
        zip_bytes = ctx.submission_zip_path.stat().st_size
        zip_sha = sha256_file(ctx.submission_zip_path)

        copied_working_zip: str | None = None
        if cfg.submission.working_zip_path:
            wpath = Path(cfg.submission.working_zip_path)
            if wpath.parent.is_dir():
                if wpath.resolve() != ctx.submission_zip_path.resolve():
                    shutil.copy2(ctx.submission_zip_path, wpath)
                copied_working_zip = str(wpath)
            else:
                notes.append(f"Skipped copying to {wpath} (parent directory does not exist)")

        outputs = {
            "template_source": template_source,
            "served_model_id": served_id,
            "adapter_sources": adapter_sources,
            "validation": validation_info,
            "submission_zip_path": str(ctx.submission_zip_path),
            "working_zip_path": copied_working_zip,
            "submission_zip_bytes": zip_bytes,
            "submission_zip_sha256": zip_sha,
        }

        ctx.update_manifest(
            {
                "served_model_id": served_id,
                "adapter_sources": adapter_sources,
                "submission_unpacked_bytes": validation_info["unpacked_size_bytes"],
                "submission_zip_path": str(ctx.submission_zip_path),
                "working_zip_path": copied_working_zip,
                "submission_zip_bytes": zip_bytes,
                "submission_zip_sha256": zip_sha,
            }
        )

        t_end = time.time()
        status = PhaseStatus(
            phase="package",
            state="ok",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            outputs=outputs,
            notes=notes,
        )
        ctx.write_status(status)
        logger.info(
            "Packaged valid submission.zip (%.2f MiB, sha256=%s...)",
            zip_bytes / (1024 * 1024),
            zip_sha[:12],
        )
        return status.to_dict()

    except Exception as exc:
        t_end = time.time()
        logger.error("Package phase failed: %s", exc)
        status = PhaseStatus(
            phase="package",
            state="failed",
            started=utc_iso(t_start),
            ended=utc_iso(t_end),
            started_ts=t_start,
            ended_ts=t_end,
            planned_deadline=utc_iso(deadline) if deadline else None,
            planned_deadline_ts=deadline,
            planned_duration_sec=round(deadline - t_start, 2) if deadline else None,
            actual_duration_sec=round(t_end - t_start, 2),
            error=str(exc),
            outputs={"adapter_sources": adapter_sources},
            notes=notes,
        )
        ctx.write_status(status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-LoRA SFT Phase 5: package")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline", type=float, default=None)
    args = parser.parse_args(argv)

    ctx = RunContext(args.run_dir)
    try:
        run_package_phase(ctx, deadline=args.deadline)
        return 0
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
