"""The Kaggle driver against stand-ins with the real harness signatures (same stand-ins as
`pipeline/multi_lora_sft/tests/test_evaluate_api.py`, whose call shapes ran on Kaggle on 2026-10-31)."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from pipeline.exp_eval.kaggle_run import main
from pipeline.multi_lora_sft.tests.test_evaluate_api import _EvalConfig, _VllmConfig


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"events": [], "configs": {}, "inflight": 0, "max_inflight": 0}

    class Manifest:
        adapters: ClassVar[dict[str, Any]] = {}

    class VllmServer:
        def __init__(self, config: _VllmConfig, adapter_manifest: Any = None) -> None:
            assert isinstance(config, _VllmConfig)
            rec["vllm"] = config
            self.process = None
            self.log_path = None

        def create_model_registry(self, aliases: Any = None, model_prefix: str | None = None,
                                  api_key: str | None = None) -> str:
            rec["aliases"] = aliases
            return "REGISTRY"

        def start(self) -> None:
            rec["events"].append("server.start")

        def stop(self) -> None:
            rec["events"].append("server.stop")

    class Evaluator:
        def __init__(self, config: _EvalConfig) -> None:
            assert isinstance(config, _EvalConfig)
            rec["configs"][Path(config.submission_dir).parent.name] = config
            rec["events"].append("evaluator.init")
            self.config = config

        async def evaluate_task(self, task: Any, task_index: int, total_tasks: int) -> Any:
            rec["inflight"] += 1
            rec["max_inflight"] = max(rec["max_inflight"], rec["inflight"])
            await asyncio.sleep(0.01)
            rec["inflight"] -= 1
            arm = Path(self.config.submission_dir).parent.name
            # The real Evaluator writes its trace under results_dir/traces.
            tdir = Path(self.config.results_dir) / "traces"
            tdir.mkdir(parents=True, exist_ok=True)
            steps = [{"source": "user", "message": "t", "extra": {"event_type": "task_prompt", "elapsed_s": 1}},
                     {"source": "agent", "llm_call_count": 1, "extra": {"event_type": "thinking", "elapsed_s": 9},
                      "tool_calls": [{"function_name": "submit_patch", "arguments": {}}],
                      "observation": {"content": json.dumps({"status": "ok", "patch_size": 5})}}]
            (tdir / f"trace_{task.instance_id}.json").write_text(json.dumps({"steps": steps}))
            win = arm.endswith("a1_skill_first") and task.instance_id.startswith("rich")
            return types.SimpleNamespace(resolved=win, agent_patch="diff --git a/x.py b/x.py\n", tool_calls=3)

    def load_tasks(path: Path) -> list[Any]:
        rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
        return [types.SimpleNamespace(instance_id=r["instance_id"], repo=r["repo"]) for r in rows]

    def mod(name: str, **attrs: Any) -> None:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    mod("adk_submission", VllmConfig=_VllmConfig, VllmServer=VllmServer,
        discover_adapters=lambda d, adapter_extensions=None: Manifest())
    mod("litellm", drop_params=False)
    mod("google")
    mod("google.adk")
    mod("google.adk.agents")
    mod("google.adk.agents.context_cache_config", ContextCacheConfig=lambda **kw: ("cache", kw))
    mod("google.adk.apps")
    mod("google.adk.apps._configs", EventsCompactionConfig=lambda **kw: ("compaction", kw))
    mod("swegemma")
    mod("swegemma.config", ALLOWED_ADAPTER_EXTENSIONS=frozenset({".safetensors"}), EvalConfig=_EvalConfig,
        build_submission_limits=lambda: ("LIMITS", "GEN"))
    mod("swegemma.evaluate", Evaluator=Evaluator)
    mod("swegemma.models", load_tasks=load_tasks)

    dev30 = json.loads((Path(__file__).resolve().parents[3] / "experiments" / "exp_016_skill_first_agent"
                        / "dev_30_ids.json").read_text())["task_ids"]
    comp = tmp_path / "comp"
    (comp / "snapshots").mkdir(parents=True)
    (comp / "tasks.jsonl").write_text("".join(json.dumps({"instance_id": t, "repo": t.rsplit("_", 1)[0]}) + "\n"
                                              for t in dev30))
    rec["comp"] = comp
    return rec


def test_driver_runs_three_arms_on_one_server_with_each_arms_own_budget(harness: dict, tmp_path: Path) -> None:
    out = tmp_path / "out"
    rc = main(["--base-model", "/models/w4a16", "--comp", str(harness["comp"]), "--out", str(out),
               "--arms", "ctrl_022,exp016_a0_pruned,exp016_a1_skill_first", "--tasks", "smoke5",
               "--concurrency", "3", "--control", "ctrl_022"])
    assert rc == 0
    ev = harness["events"]
    assert ev.count("server.start") == 1 and ev[-1] == "server.stop"  # no adapters -> one shared server
    assert ev.index("evaluator.init") < ev.index("server.start")
    assert harness["max_inflight"] == 3  # the shared slot pool is really used

    v = harness["vllm"]
    assert v.enable_lora is False and v.tensor_parallel_size == 4 and v.max_model_len == 32768
    assert v.tool_call_parser == "gemma4" and "--max-num-seqs" in v.extra_args
    assert harness["aliases"] == ["gemma-4-31b-it-qat-w4a16-ct"]

    cfg = harness["configs"]
    assert cfg["ctrl_022"].max_tool_calls == 28 and cfg["exp016_a1_skill_first"].max_tool_calls == 30
    assert all(c.max_time_minutes == 8 and c.timeout_seconds == 240 and c.max_turns == 80 for c in cfg.values())
    assert all(c.sandbox == "subprocess" and Path(c.results_dir).name == "harness" for c in cfg.values())

    card = json.loads((out / "scorecard.json").read_text())
    assert card["control"] == "ctrl_022"
    a1 = next(c for c in card["comparisons"] if c["b"] == "exp016_a1_skill_first")
    assert a1["common_tasks"] == 5 and a1["only_b"] == ["rich_2725"]
    assert all(a["summary"]["end_reasons"] == {"submitted": 5} for a in card["arms"])
    assert (out / "scorecard.md").is_file() and (out / "exp016_a1_skill_first" / "submission.zip").is_file()
    assert json.loads((out / "run_config.json").read_text())["task_ids"][0] == "fastapi_11194"


def test_driver_refuses_unknown_tasks_before_starting_vllm(harness: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in tasks.jsonl"):
        main(["--base-model", "/m", "--comp", str(harness["comp"]), "--out", str(tmp_path / "o"),
              "--arms", "ctrl_022", "--tasks", "nope_1,nope_2"])
    assert "server.start" not in harness["events"]
