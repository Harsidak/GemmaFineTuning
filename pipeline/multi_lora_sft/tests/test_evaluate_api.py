"""The eval phase against stand-ins for swegemma / adk_submission with the real signatures.

Regression for the 2026-10-31 Kaggle run: evaluate.py guessed the swegemma API, a kwargs filter silently dropped every
argument, and EvalConfig() failed after a 5-minute vLLM load. The same filter also dropped `lora_modules`, so adapters
were never mounted. The stand-ins below copy the signatures used by the official "Getting Started" notebook and the
adk_submission 0.2.13 VllmConfig fields; unknown keyword arguments raise like the real classes do."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from pipeline.multi_lora_sft.config import load_config
from pipeline.multi_lora_sft.phases.evaluate import run_eval_phase
from pipeline.multi_lora_sft.runctx import RunContext


@dataclass
class _VllmConfig:  # adk_submission 0.2.13 VllmConfig fields
    model: str
    port: int = 8000
    host: str = "127.0.0.1"
    tool_call_parser: str | None = "hermes"
    reasoning_parser: str | None = None
    reasoning_config: Any = None
    default_chat_template_kwargs: Any = None
    limit_mm_per_prompt: str | None = None
    speculative_config: Any = None
    kv_cache_dtype: str | None = None
    max_model_len: int = 32768
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    enable_auto_tool_choice: bool = True
    enable_lora: bool = True
    max_loras: int = 1
    max_lora_rank: int = 128
    scheduler_reserve_full_isl: bool = False
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 600
    tensor_parallel_size: int = 1
    health_check_interval: float = 1.0
    nvidia_lib_dir: str = "/usr/local/nvidia/lib64"


@dataclass
class _EvalConfig:  # keyword set of the official notebook's EvalConfig(...) call; the first three are required
    tasks_path: Path
    results_dir: Path
    models: Any
    snapshots_dir: Path | None = None
    submission_dir: Path | None = None
    sandbox: str = "docker"
    timeout_seconds: int = 300
    verification_timeout_seconds: int = 300
    max_time_minutes: float = 60.0
    max_tool_calls: int | None = None
    max_turns: int | None = None
    limits: Any = None
    generation_constraints: Any = None
    adapter_manifest: Any = None
    context_cache_config: Any = None
    events_compaction_config: Any = None
    graph_dir: str | None = None
    embeddings_dir: str | None = None
    wheels_dir: Path | None = None
    verbose: bool = True


@pytest.fixture
def fake_harness(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rec: dict[str, Any] = {"events": []}

    class Manifest:
        def __init__(self, adapters: dict[str, Any]) -> None:
            self.adapters = adapters

    def discover_adapters(submission_dir: str, adapter_extensions: Any = None) -> Manifest:
        root = Path(submission_dir) / "adapters"
        found = {d.name: d for d in sorted(root.iterdir())} if root.is_dir() else {}
        return Manifest({n: types.SimpleNamespace(path=p, rank=8) for n, p in found.items()})

    class VllmServer:
        def __init__(self, config: _VllmConfig, adapter_manifest: Any = None, lora_modules: Any = None) -> None:
            assert isinstance(config, _VllmConfig)
            self.config, self.adapter_manifest = config, adapter_manifest
            self.process = None
            self.log_path = None
            rec["server"] = self

        def create_model_registry(self, aliases: Any = None, model_prefix: str | None = "hosted_vllm/",
                                  api_key: str | None = "EMPTY") -> str:
            rec["registry_args"] = (aliases, model_prefix)
            return "REGISTRY"

        def start(self) -> None:
            rec["events"].append("server.start")

        def stop(self) -> None:
            rec["events"].append("server.stop")

    class Evaluator:
        def __init__(self, config: _EvalConfig) -> None:
            assert isinstance(config, _EvalConfig)
            rec["eval_config"] = config
            rec["events"].append("evaluator.init")

        async def evaluate_task(self, task: Any, task_index: int, total_tasks: int) -> Any:
            rec["events"].append(f"task:{task.instance_id}:{task_index}/{total_tasks}")
            return types.SimpleNamespace(resolved=task_index == 1, agent_patch="diff --git a/x b/x\n",
                                         test_exit_code=0, tool_calls=7, duration_seconds=1.0)

    def load_tasks(path: Path) -> list[Any]:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return [types.SimpleNamespace(instance_id=r["instance_id"], repo=r["repo"]) for r in rows]

    def mod(name: str, **attrs: Any) -> None:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    mod("adk_submission", VllmConfig=_VllmConfig, VllmServer=VllmServer, discover_adapters=discover_adapters)
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
    return rec


def test_eval_phase_uses_official_api_and_builds_evaluator_before_vllm(tmp_path: Path, fake_harness: dict) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"instance_id": f"t{i}", "repo": f"r{i}"}) + "\n" for i in range(4)),
                     encoding="utf-8")
    (tmp_path / "snapshots").mkdir()
    cfg = load_config(set_overrides=[
        f"run.output_root={(tmp_path / 'runs').as_posix()}",
        "run.run_id=eval_api",
        f"data.tasks_file={tasks.as_posix()}",
        f"eval.snapshots_dir={(tmp_path / 'snapshots').as_posix()}",
        "model.served_path=/models/gemma-4-31b-it-qat-w4a16-ct/2",
        "model.served_model_id=gemma-4-31b-it-qat-w4a16-ct",
    ])
    ctx = RunContext.create(cfg, t0=1000.0)
    lora = ctx.submission_dir / "adapters" / "main_lora"
    lora.mkdir(parents=True)
    (lora / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
    (ctx.submission_dir / "eval_config.yaml").write_text(
        "evaluation:\n  timeout_seconds: 120\n  max_tool_calls: 50\n  max_time_minutes: 60.0\n  max_turns: 500\n",
        encoding="utf-8")

    import time
    st = run_eval_phase(ctx, cfg=cfg, deadline=time.time() + 3600)
    assert st["state"] == "ok", st.get("error")

    ev = fake_harness["events"]
    assert ev.index("evaluator.init") < ev.index("server.start")  # API mismatches surface before the model loads
    assert [e for e in ev if e.startswith("task:")] == ["task:t0:1/3", "task:t1:2/3", "task:t2:3/3"]
    assert ev[-1] == "server.stop"

    srv = fake_harness["server"]
    assert srv.adapter_manifest is not None and set(srv.adapter_manifest.adapters) == {"main_lora"}
    assert srv.config.enable_lora and srv.config.max_lora_rank == 16
    assert "--max-num-seqs" in srv.config.extra_args
    assert fake_harness["registry_args"] == (["gemma-4-31b-it-qat-w4a16-ct"], "openai/")

    ec = fake_harness["eval_config"]
    assert ec.models == "REGISTRY" and ec.sandbox == "subprocess" and ec.submission_dir == ctx.submission_dir
    assert ec.timeout_seconds == 120 and ec.max_tool_calls == 50 and ec.max_turns == 500
    assert ec.events_compaction_config == ("compaction", {"compaction_interval": 15, "overlap_size": 2,
                                                          "token_threshold": 14336, "event_retention_size": 5})
    assert 2.0 <= ec.max_time_minutes < 60.0  # capped so 3 tasks fit the 1 h left

    rows = st["outputs"]["tasks"]
    assert [r["result"] for r in rows] == ["PASS", "FAIL", "FAIL"]
    assert rows[0]["patch_applied"] is True and rows[0]["tool_calls"] == 7
