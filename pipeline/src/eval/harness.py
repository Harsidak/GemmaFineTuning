import os
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LocalEvaluator:
    """Evaluates agent patches on benchmark tasks and calculates Pass@1 accuracy."""

    def __init__(self, workspace_root: Optional[str | Path] = None):
        self.workspace_root = Path(workspace_root or ".")

    def evaluate_patch(
        self,
        repo_dir: str | Path,
        patch_diff: str,
        test_command: str = "pytest"
    ) -> Dict[str, Any]:
        """Applies a patch to the repository and executes the unit test command."""
        repo_path = Path(repo_dir)
        patch_file = repo_path / "temp_eval.patch"

        try:
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(patch_diff)

            # Apply patch via git apply
            apply_res = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if apply_res.returncode != 0:
                return {
                    "passed": False,
                    "error": f"Failed to apply patch: {apply_res.stderr.strip()}"
                }

            # Run validation test suite
            test_res = subprocess.run(
                test_command.split(),
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=180
            )

            passed = (test_res.returncode == 0)
            return {
                "passed": passed,
                "stdout": test_res.stdout,
                "stderr": test_res.stderr
            }

        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test execution timed out."}
        except Exception as e:
            return {"passed": False, "error": str(e)}
        finally:
            # Revert repository state
            if patch_file.exists():
                patch_file.unlink()
            subprocess.run(["git", "checkout", "."], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)

    def calculate_pass_at_1(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        pass_rate = (passed / total) if total > 0 else 0.0

        return {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_at_1": round(pass_rate, 4),
            "pass_rate_pct": round(pass_rate * 100, 2)
        }
