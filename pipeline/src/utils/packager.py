import os
import zipfile
import logging
from pathlib import Path
from typing import Tuple, List
import yaml

logger = logging.getLogger(__name__)

# Support official ADK custom tags such as !include
class ADKYamlLoader(yaml.SafeLoader):
    pass

ADKYamlLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))

def validate_submission_directory(directory: str | Path) -> Tuple[bool, List[str]]:
    """Validates that a submission directory complies with ADK competition rules:

    1. Root must contain 'agent.yaml'.
    2. No arbitrary Python execution scripts (*.py) in root.
    3. Proper YAML parsing of agent.yaml (supports !include tags).
    4. Base model consistency check across all agent definitions.
    """
    dir_path = Path(directory)
    errors = []

    if not dir_path.exists():
        return False, [f"Directory does not exist: {dir_path}"]

    agent_yaml = dir_path / "agent.yaml"
    if not agent_yaml.exists():
        errors.append("Missing required root 'agent.yaml'.")

    # Prohibit custom python entrypoints
    py_files = list(dir_path.glob("*.py"))
    if py_files:
        errors.append(f"Prohibited Python entrypoints found in submission root: {[p.name for p in py_files]}")

    # Validate agent.yaml schema
    root_model_name = None
    if agent_yaml.exists():
        try:
            with open(agent_yaml, "r", encoding="utf-8") as f:
                root_data = yaml.load(f, Loader=ADKYamlLoader)
            if not isinstance(root_data, dict):
                errors.append("Root agent.yaml is not a valid YAML mapping.")
            else:
                model_val = root_data.get("model")
                if isinstance(model_val, dict):
                    root_model_name = model_val.get("name")
                elif isinstance(model_val, str):
                    root_model_name = model_val
                
                if not root_model_name:
                    errors.append("agent.yaml missing 'model' attribute.")
        except Exception as e:
            errors.append(f"Failed to parse agent.yaml: {e}")

    # Check sub-agents for model consistency
    sub_agents_dir = dir_path / "sub_agents"
    if sub_agents_dir.exists() and root_model_name:
        for sub_agent_file in sub_agents_dir.glob("*.yaml"):
            try:
                with open(sub_agent_file, "r", encoding="utf-8") as f:
                    sub_data = yaml.load(f, Loader=ADKYamlLoader)
                sub_model_val = sub_data.get("model")
                if isinstance(sub_model_val, dict):
                    sub_model = sub_model_val.get("name")
                elif isinstance(sub_model_val, str):
                    sub_model = sub_model_val
                else:
                    sub_model = None

                if sub_model and sub_model != root_model_name:
                    errors.append(
                        f"Model mismatch in {sub_agent_file.name}: "
                        f"Targets '{sub_model}', but root targets '{root_model_name}'. "
                        "All agents and sub-agents must use the identical base model."
                    )
            except Exception as e:
                errors.append(f"Failed to parse {sub_agent_file.name}: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors


def package_submission_archive(
    source_dir: str | Path,
    output_zip_path: str | Path = "submission.zip"
) -> Path:
    """Packages validated submission files into submission.zip."""
    src = Path(source_dir)
    out_zip = Path(output_zip_path)

    is_valid, errors = validate_submission_directory(src)
    if not is_valid:
        raise ValueError(f"Submission directory validation failed: {errors}")

    if out_zip.exists():
        out_zip.unlink()

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in src.rglob("*"):
            if file.is_file():
                # Avoid hidden files
                if any(part.startswith(".") for part in file.parts):
                    continue
                arcname = file.relative_to(src)
                zf.write(file, arcname)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully packaged {out_zip} ({size_mb:.2f} MB)")
    return out_zip
