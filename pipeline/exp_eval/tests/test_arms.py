"""Arm overlays: builds, one-variable manifests, fidelity to the packaged zips and the competition rule checks."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from pipeline.exp_eval.arms import (
    DEFAULT_ARMS_ROOT,
    REPO_ROOT,
    ArmError,
    Include,
    build_arm,
    check_submission_rules,
    dump_yaml,
    list_arms,
    load_yaml,
)
from pipeline.src.utils.packager import validate_submission_directory


def test_all_committed_arms_build_and_pass_the_rules(tmp_path: Path) -> None:
    arms = list_arms()
    assert {"ctrl_022", "exp016_a0_pruned", "exp016_a1_skill_first"} <= set(arms)
    for arm_id in arms:
        built = build_arm(arm_id, tmp_path / arm_id)
        ok, errs = validate_submission_directory(built.submission_dir)  # the repo's own packager check too
        assert ok, (arm_id, errs)
        assert built.manifest["rule_warnings"] == [], (arm_id, built.manifest["rule_warnings"])
        assert built.zip_path.is_file()


def test_each_exp016_arm_changes_only_what_it_declares(tmp_path: Path) -> None:
    a0 = build_arm("exp016_a0_pruned", tmp_path / "a0").manifest
    assert a0["chain"] == ["ctrl_022", "exp016_a0_pruned"]
    assert a0["changed_vs_base"] == ["agent.yaml", "prompts/system.md"]
    assert a0["eval_config"]["max_tool_calls"] == 28  # budget untouched: tool pruning is the only variable

    a1 = build_arm("exp016_a1_skill_first", tmp_path / "a1").manifest
    assert a1["changed_vs_base"] == [
        "agent.yaml", "eval_config.yaml", "prompts/system.md", "skills/swe-kit/SKILL.md",
        "skills/swe-kit/scripts/locate.py", "skills/swe-kit/scripts/read_symbol.py",
        "skills/swe-kit/scripts/verify_and_audit.py",
    ]
    assert a1["eval_config"]["max_tool_calls"] == 30
    assert not any("__pycache__" in f or f.endswith(".pyc") for f in a1["files"])


def test_a0_prunes_exactly_the_three_graph_tools(tmp_path: Path) -> None:
    ctrl = load_yaml(build_arm("ctrl_022", tmp_path / "c").submission_dir / "agent.yaml")
    a0_dir = build_arm("exp016_a0_pruned", tmp_path / "a0").submission_dir
    a0 = load_yaml(a0_dir / "agent.yaml")
    assert set(ctrl["tools"]) - set(a0["tools"]) == {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
    assert {k: v for k, v in ctrl.items() if k != "tools"} == {k: v for k, v in a0.items() if k != "tools"}
    ctrl_prompt = (tmp_path / "c" / "submission" / "prompts" / "system.md").read_text(encoding="utf-8").splitlines()
    a0_prompt = (a0_dir / "prompts" / "system.md").read_text(encoding="utf-8").splitlines()
    assert [ln for ln in ctrl_prompt if ln not in a0_prompt] == [
        "- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful."
    ]


def test_builds_are_byte_reproducible(tmp_path: Path) -> None:
    h1 = build_arm("exp016_a1_skill_first", tmp_path / "x").manifest["zip_sha256"]
    h2 = build_arm("exp016_a1_skill_first", tmp_path / "y").manifest["zip_sha256"]
    assert h1 == h2


@pytest.mark.skipif(not (REPO_ROOT / "kaggle" / "submission.zip").is_file(), reason="0.22 zip not present (gitignored)")
def test_ctrl_is_byte_identical_to_the_022_submission(tmp_path: Path) -> None:
    sub = build_arm("ctrl_022", tmp_path / "c").submission_dir
    with zipfile.ZipFile(REPO_ROOT / "kaggle" / "submission.zip") as zf:
        names = sorted(zf.namelist())
        assert names == sorted(p.relative_to(sub).as_posix() for p in sub.rglob("*") if p.is_file())
        for n in names:
            assert zf.read(n) == (sub / n).read_bytes(), n


A1_ZIP = REPO_ROOT / "runs" / "exp_016" / "submission_A1_skill_first.zip"


@pytest.mark.skipif(not A1_ZIP.is_file(), reason="packaged EXP-016 A1 zip not present (runs/ is gitignored)")
def test_a1_matches_the_packaged_exp016_zip(tmp_path: Path) -> None:
    sub = build_arm("exp016_a1_skill_first", tmp_path / "a1").submission_dir
    with zipfile.ZipFile(A1_ZIP) as zf:
        assert sorted(zf.namelist()) == sorted(p.relative_to(sub).as_posix() for p in sub.rglob("*") if p.is_file())
        for n in zf.namelist():
            if n == "skills/swe-kit/scripts/locate.py":
                continue  # deliberately changed: no stdin fallback (see arm.yaml)
            if n.endswith(".yaml"):
                assert yaml.load(zf.read(n), Loader=yaml.BaseLoader) == yaml.load(
                    (sub / n).read_bytes(), Loader=yaml.BaseLoader), n
            else:
                assert zf.read(n) == (sub / n).read_bytes(), n


def test_include_tags_survive_a_rewrite_unquoted() -> None:
    data = {"model": "m", "instruction": Include("prompts/system.md"), "tools": ["run_command"]}
    text = dump_yaml(data)
    assert "instruction: !include prompts/system.md\n" in text
    from pipeline.src.utils.packager import ADKYamlLoader

    assert yaml.load(text, Loader=ADKYamlLoader)["instruction"] == "prompts/system.md"


# ----------------------------------------------------------------------------------------------- spec validation


def _arms_root_with(tmp_path: Path, specs: dict[str, str]) -> Path:
    root = tmp_path / "arms"
    shutil.copytree(DEFAULT_ARMS_ROOT / "ctrl_022", root / "ctrl_022")
    for arm_id, text in specs.items():
        (root / arm_id).mkdir(parents=True)
        (root / arm_id / "arm.yaml").write_text(text, encoding="utf-8")
    return root


def test_a_declared_change_that_changes_nothing_is_rejected(tmp_path: Path) -> None:
    root = _arms_root_with(tmp_path, {"noop": (
        "id: noop\nbase: ctrl_022\nchanges:\n  yaml_set:\n    eval_config.yaml:\n"
        "      evaluation.max_tool_calls: 28\n")})
    with pytest.raises(ArmError, match="unchanged"):
        build_arm("noop", tmp_path / "out", arms_root=root)


def test_overlays_must_declare_changes_and_full_arms_must_not(tmp_path: Path) -> None:
    root = _arms_root_with(tmp_path, {"empty": "id: empty\nbase: ctrl_022\n"})
    with pytest.raises(ArmError, match="at least one change"):
        build_arm("empty", tmp_path / "o1", arms_root=root)
    with pytest.raises(ArmError, match="no arm"):
        build_arm("missing", tmp_path / "o2", arms_root=root)


def test_rule_breaking_overlay_fails_the_build(tmp_path: Path) -> None:
    root = _arms_root_with(tmp_path, {"bad_tool": (
        "id: bad_tool\nbase: ctrl_022\nchanges:\n  yaml_set:\n    agent.yaml:\n"
        "      tools: [run_command, code_graph_search]\n")})
    with pytest.raises(ArmError, match="code_graph_search"):
        build_arm("bad_tool", tmp_path / "out", arms_root=root)


def _write_sub(tmp_path: Path, agent: str, files: dict[str, str]) -> Path:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "agent.yaml").write_text(agent, encoding="utf-8")
    for rel, text in files.items():
        (sub / rel).parent.mkdir(parents=True, exist_ok=True)
        (sub / rel).write_text(text, encoding="utf-8")
    return sub


GOOD_AGENT = ("name: a\nmodel: gemma-4-31b-it-qat-w4a16-ct\ninstruction: !include prompts/system.md\n"
              "tools: [run_command, submit_patch]\ngenerate_content_config: !include configs/sampling.yaml\n")


def test_rule_checker_flags_each_violation(tmp_path: Path) -> None:
    sub = _write_sub(tmp_path, GOOD_AGENT.replace("w4a16-ct", "w4a16-ct-typo") + "adapter: main_lora\n", {
        "prompts/system.md": "You have 30 tool calls and 8 minutes. Use `read_file`.\n",
        "configs/sampling.yaml": "max_output_tokens: 40000\nthinking_config:\n  thinking_budget: 4096\n",
        "eval_config.yaml": "evaluation:\n  max_tool_calls: 28\n  max_llm_calls: 9\n",
        "helper.py": "print(1)\n",
        "skills/x/scripts/cache.pyc": "",
        "notes.ipynb": "{}",
    })
    errors, warnings = check_submission_rules(sub)
    text = " | ".join(errors)
    assert "one allowed base model" in text
    assert "helper.py: .py files are only allowed inside skills/" in text
    assert "notes.ipynb: extension .ipynb is not allowed" in text
    assert "max_output_tokens=40000" in text
    assert "max_llm_calls" in text
    assert "adapter 'main_lora'" in text
    assert not any("cache.pyc" in e for e in errors)  # byte-code is ignored, never shipped
    assert any("[30] tool calls" in w and "max_tool_calls=28" in w for w in warnings)
    assert any("read_file" in w for w in warnings)


def test_rule_checker_accepts_a_clean_submission(tmp_path: Path) -> None:
    sub = _write_sub(tmp_path, GOOD_AGENT, {
        "prompts/system.md": "28 tool calls and 8 minutes. Call `submit_patch` last.\n",
        "configs/sampling.yaml": "max_output_tokens: 8192\n",
        "eval_config.yaml": "evaluation:\n  max_tool_calls: 28\n  max_time_minutes: 8\n",
    })
    assert check_submission_rules(sub) == ([], [])


def test_zip_contains_no_bytecode_even_if_the_source_has_it(tmp_path: Path) -> None:
    built = build_arm("exp016_a1_skill_first", tmp_path / "a1")
    with zipfile.ZipFile(built.zip_path) as zf:
        assert not [n for n in zf.namelist() if "__pycache__" in n or n.endswith(".pyc")]
    assert built.manifest["zip_sha256"] == hashlib.sha256(built.zip_path.read_bytes()).hexdigest()
