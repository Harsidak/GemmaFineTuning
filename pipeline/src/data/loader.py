import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_tasks(filepath: str | Path) -> List[Dict[str, Any]]:
    """Loads competition evaluation tasks from a JSONL file.

    Expected task keys:
    - task_id (str): Unique identifier (e.g. 'fastapi_15661')
    - repo (str): Repository name
    - base_commit (str): Git commit hash
    - issue_description (str): Problem statement
    - hints_text (str, optional): Hints provided with the issue
    - test_patch (str, optional): Ground-truth test patch
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Tasks file not found at: {path}")
        return []

    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding line {idx} in {path}: {e}")

    logger.info(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


def load_trajectories(filepath: str | Path, successful_only: bool = True) -> List[Dict[str, Any]]:
    """Loads agent conversation trajectories for SFT fine-tuning.

    Each trajectory typically contains:
    - task_id: Identifier
    - repo: Repository
    - messages: List of dicts (role, content, tool_calls, tool_call_id, etc.)
    - resolved: bool (whether the trajectory resulted in a passing patch)
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Trajectories file not found at: {path}")
        return []

    trajectories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            if successful_only and not entry.get("resolved", True):
                continue
            trajectories.append(entry)

    logger.info(f"Loaded {len(trajectories)} trajectories (successful_only={successful_only}) from {path}")
    return trajectories
