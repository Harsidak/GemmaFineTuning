"""Unit and packaging tests for EXP-016 (`swe-kit` ADK skill + `pipeline/submission_template`)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.package import (
    run_package_phase,
    validate_staged_submission,
)
from pipeline.multi_lora_sft.runctx import RunContext


def _load_skill_module(script_name: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "data" / "Skills" / "swe-kit" / "scripts" / script_name
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(f"swe_kit_{script_path.stem}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_git_repo(ws: Path) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=ws, check=True)


def _commit_all(ws: Path, msg: str = "baseline") -> None:
    subprocess.run(["git", "add", "."], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ws, check=True)


def test_locate_ranks_docs_src_and_maps_parent_tests(tmp_path: Path) -> None:
    locate = _load_skill_module("locate.py")
    ws = tmp_path / "repo"
    (ws / "fastapi" / "dependencies").mkdir(parents=True)
    (ws / "docs_src" / "security").mkdir(parents=True)
    (ws / "tests").mkdir(parents=True)

    (ws / "fastapi" / "dependencies" / "utils.py").write_text(
        "def solve_dependencies(*, dependant, dependency_overrides_provider=None):\n"
        "    return dependant, dependency_overrides_provider\n",
        encoding="utf-8",
    )
    (ws / "fastapi" / "routing.py").write_text(
        "class APIRoute:\n    def get_route_handler(self):\n        return None\n",
        encoding="utf-8",
    )
    (ws / "docs_src" / "security" / "tutorial001.py").write_text(
        "from fastapi.security import OAuth2PasswordBearer\n"
        "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')\n",
        encoding="utf-8",
    )
    (ws / "tests" / "test_dependencies.py").write_text(
        "from fastapi.dependencies.utils import solve_dependencies\n\n"
        "def test_solve_deps():\n    assert solve_dependencies(dependant='ok')[0] == 'ok'\n",
        encoding="utf-8",
    )
    (ws / "tests" / "test_tutorial_security.py").write_text(
        "from docs_src.security import tutorial001\n\n"
        "def test_oauth2():\n    assert tutorial001.oauth2_scheme is not None\n",
        encoding="utf-8",
    )

    # 1. Generic stem `utils.py` under `fastapi/dependencies/` resolves & maps to `tests/test_dependencies.py`
    res_dep = locate.rank_workspace(
        ws,
        query="Bug in solve_dependencies when dependency_overrides_provider is passed\n\n### AI Disclaimer\nGenerated help",
    )
    assert res_dep["status"] == "ok"
    assert res_dep["top_candidates"][0]["file"] == "fastapi/dependencies/utils.py"
    assert any(s["name"] == "solve_dependencies" for s in res_dep["top_candidates"][0]["symbols"])
    assert "tests/test_dependencies.py" in res_dep["recommended_tests"]

    # 2. `docs_src/` tutorial files are indexed (never ignored) and ranked first for tutorial issues
    res_doc = locate.rank_workspace(
        ws,
        query="Fix OAuth2PasswordBearer tokenUrl example in docs_src/security/tutorial001.py",
    )
    assert res_doc["status"] == "ok"
    assert res_doc["top_candidates"][0]["file"] == "docs_src/security/tutorial001.py"
    assert "tests/test_tutorial_security.py" in res_doc["recommended_tests"]

    # 3. Omnivorous CLI normalization (JSON dict string + shlex string)
    assert locate._normalize_argv(['{"query": "solve_dependencies", "top_k": 3}']) == [
        "--query",
        "solve_dependencies",
        "--top-k",
        "3",
    ]
    assert locate._normalize_argv(['--query "solve_dependencies bug" --top-k 4']) == [
        "--query",
        "solve_dependencies bug",
        "--top-k",
        "4",
    ]


def test_read_symbol_extracts_methods_decorators_enclosing_and_truncates(tmp_path: Path) -> None:
    read_symbol = _load_skill_module("read_symbol.py")
    ws = tmp_path / "repo"
    (ws / "pkg").mkdir(parents=True)

    big_body = "\n".join(f"        x_{i} = {i}" for i in range(150))
    (ws / "pkg" / "service.py").write_text(
        "def helper(val: int) -> int:\n"
        "    return val + 1\n\n"
        "class Router:\n"
        "    @classmethod\n"
        "    async def dispatch(cls, path: str) -> str:\n"
        "        if not path.startswith('/'):\n"
        "            raise ValueError('path must start with slash')\n"
        "        return path\n\n"
        "    def huge_method(self):\n"
        f"{big_body}\n"
        "        return x_149\n",
        encoding="utf-8",
    )

    # Exact Class.method with decorator included in start_line
    sym_res = read_symbol.inspect_file(ws, "pkg/service.py", symbol="Router.dispatch")
    assert sym_res["status"] == "ok"
    assert sym_res["kind"] == "async_def"
    assert sym_res["start_line"] == 5  # @classmethod line
    assert sym_res["end_line"] == 9
    assert "@classmethod" in sym_res["code"]
    assert "path must start with slash" in sym_res["code"]

    # Shorthand `pkg/service.py:8` resolves enclosing `Router.dispatch`
    ln_res = read_symbol.inspect_file(ws, "pkg/service.py:8")
    assert ln_res["status"] == "ok"
    assert ln_res["enclosing_symbol"] == "Router.dispatch"
    assert "8* |" in ln_res["code"]

    # Giant method (>120 lines) is cleanly capped with head + tail
    huge_res = read_symbol.inspect_file(ws, "pkg/service.py", symbol="huge_method")
    assert huge_res["status"] == "ok"
    assert huge_res["truncated"] is True
    assert "omitted; pass --range" in huge_res["code"]
    assert "return x_149" in huge_res["code"]


def test_verify_and_audit_blocks_leaks_stripped_only_syntax_and_runs_pytest(tmp_path: Path) -> None:
    auditor = _load_skill_module("verify_and_audit.py")
    ws = tmp_path / "repo"
    (ws / "mylib").mkdir(parents=True)
    (ws / "tests").mkdir(parents=True)

    (ws / "pyproject.toml").write_text("[project]\nname = 'mylib'\n", encoding="utf-8")
    (ws / "mylib" / "calc.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    (ws / "tests" / "test_calc.py").write_text(
        "import pytest\n"
        "from mylib.calc import divide\n\n"
        "def test_divide_zero_msg():\n"
        "    with pytest.raises(ValueError, match='divisor cannot be zero'):\n"
        "        divide(4, 0)\n",
        encoding="utf-8",
    )
    _init_git_repo(ws)
    _commit_all(ws)

    # Case 1: Clean unmodified tree -> blocked
    r_clean = auditor.verify_and_audit(ws, no_tests=True)
    assert r_clean["ready_to_submit"] is False
    assert any("EMPTY" in e for e in r_clean["blocking_errors"])

    # Case 2: Only protected files (`tests/` + `pyproject.toml`) modified -> HARD BLOCKED
    (ws / "pyproject.toml").write_text("[project]\nname = 'tampered'\n", encoding="utf-8")
    (ws / "tests" / "test_calc.py").write_text("def test_divide_zero_msg():\n    assert True\n", encoding="utf-8")
    r_prot = auditor.verify_and_audit(ws, no_tests=True)
    assert r_prot["ready_to_submit"] is False
    assert r_prot["kept_source_files"] == []
    assert set(r_prot["stripped_protected_files"]) == {"pyproject.toml", "tests/test_calc.py"}
    subprocess.run(["git", "checkout", "--", "pyproject.toml", "tests/test_calc.py"], cwd=ws, check=True)

    # Case 3: Untracked scratch script `/workspace/repro.py` -> caught on BOTH 1st AND 2nd call (post `git add -N .`)
    (ws / "repro.py").write_text("print('reproducing')\n", encoding="utf-8")
    (ws / "mylib" / "calc.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('wrong message')\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    r_leak1 = auditor.verify_and_audit(ws, no_tests=True)
    r_leak2 = auditor.verify_and_audit(ws, no_tests=True)
    assert r_leak1["ready_to_submit"] is False
    assert r_leak2["ready_to_submit"] is False
    assert any("repro.py" in e for e in r_leak2["blocking_errors"])
    (ws / "repro.py").unlink()
    subprocess.run(["git", "reset", "-q"], cwd=ws, check=True)

    # Case 4: SyntaxError in source -> blocked with line number
    (ws / "mylib" / "calc.py").write_text("def divide(a, b):\n    if b == 0\n        return 0\n", encoding="utf-8")
    r_syn = auditor.verify_and_audit(ws, no_tests=True)
    assert r_syn["ready_to_submit"] is False
    assert any("SyntaxError" in e and ":2:" in e for e in r_syn["blocking_errors"])

    # Case 5: Valid syntax, wrong message -> targeted pytest runs automatically and blocks on failure
    (ws / "mylib" / "calc.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('wrong message')\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    r_fail = auditor.verify_and_audit(ws, no_tests=False)
    assert r_fail["ready_to_submit"] is False
    assert r_fail["test_report"]["status"] == "failed"

    # Case 6: Exact spec fix -> targeted pytest passes and `ready_to_submit` is True
    (ws / "mylib" / "calc.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('divisor cannot be zero')\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    r_ok = auditor.verify_and_audit(ws, no_tests=False)
    assert r_ok["ready_to_submit"] is True, r_ok
    assert r_ok["kept_source_files"] == ["mylib/calc.py"]
    assert r_ok["test_report"]["status"] == "passed"


@pytest.mark.parametrize("with_skills", [True, False])
def test_package_exp016_template_builds_valid_bare_submission_arms(
    tmp_path: Path, repo_root: Path, with_skills: bool
) -> None:
    tpl_dir = repo_root / "pipeline" / "submission_template"
    skills_override = "submission.skills=[swe-kit]" if with_skills else "submission.skills=[]"
    cfg = load_config(
        set_overrides=[
            f"run.output_root={(tmp_path / 'runs').as_posix()}",
            f"run.run_id=exp016_arm_{int(with_skills)}",
            "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
            f"submission.template_dir={tpl_dir.as_posix()}",
            "submission.write_generation_config=false",
            "submission.write_eval_config=false",
            skills_override,
            "submission.working_zip_path=null",
        ]
    )
    ctx = RunContext.create(cfg, t0=1000.0)
    cfg.save_yaml(ctx.resolved_config_path)

    st = run_package_phase(ctx, cfg=cfg)
    assert st["state"] == "ok", st.get("error")
    val = validate_staged_submission(ctx.submission_dir, expected_served_model="gemma-4-31b-it-qat-w4a16-ct")
    assert val["present_adapters"] == []
    assert val["referenced_adapters"] == []

    agent_txt = (ctx.submission_dir / "agent.yaml").read_text(encoding="utf-8")
    assert "adapter:" not in agent_txt
    assert "search_similar_code" not in agent_txt
    assert "get_code_neighbors" not in agent_txt
    assert "get_code_subgraph" not in agent_txt

    with zipfile.ZipFile(ctx.submission_zip_path) as zf:
        names = set(zf.namelist())
    assert {"agent.yaml", "prompts/system.md", "configs/sampling.yaml", "eval_config.yaml"} <= names
    assert all("\\" not in n for n in names)

    if with_skills:
        assert "skills:\n  - skills/swe-kit\n" in agent_txt
        assert {
            "skills/swe-kit/SKILL.md",
            "skills/swe-kit/scripts/locate.py",
            "skills/swe-kit/scripts/read_symbol.py",
            "skills/swe-kit/scripts/verify_and_audit.py",
        } <= names
    else:
        assert "skills:" not in agent_txt
        assert not any(n.startswith("skills/") for n in names)


def test_dev30_split_has_exact_30_unique_official_ids(repo_root: Path) -> None:
    dev30_path = repo_root / "experiments" / "exp_016_skill_first_agent" / "dev_30_ids.json"
    data = json.loads(dev30_path.read_text(encoding="utf-8"))
    ids = data["task_ids"]
    assert data["count"] == 30
    assert len(ids) == 30
    assert len(set(ids)) == 30
