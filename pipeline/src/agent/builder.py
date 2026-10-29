"""Builds the declarative submission bundle matching the official Google ADK swegemma schema."""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .prompts import ROOT_SYSTEM_PROMPT, LOCALIZER_PROMPT

class ADKSubmissionBuilder:
    """Builds the declarative submission directory layout adhering to official ADK swegemma specifications."""

    def __init__(
        self,
        base_model_name: str = "gemma-4-31b-it-qat-w4a16-ct",
        output_dir: str | Path = "build/submission_staging"
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)

    def prepare_scaffold(self) -> Path:
        """Creates the required staging folder structure."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        (self.output_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sub_agents").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "adapters" / "main_lora").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "adapters" / "tool_lora").mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def write_prompts(self):
        """Writes prompt markdown files matching the official format."""
        prompts_dir = self.output_dir / "prompts"
        with open(prompts_dir / "system.md", "w", encoding="utf-8") as f:
            f.write(ROOT_SYSTEM_PROMPT.strip())

        with open(prompts_dir / "analyzer.md", "w", encoding="utf-8") as f:
            f.write(LOCALIZER_PROMPT.strip())

    def write_sub_agents(self, tool_adapter: Optional[str] = "tool_lora"):
        """Writes declarative sub-agent YAML configurations."""
        sub_agents_dir = self.output_dir / "sub_agents"

        # Code Analyzer sub-agent
        analyzer_cfg = {
            "name": "code_analyzer_agent",
            "description": "Analyzes repository source files and symbol graphs to locate root causes.",
            "model": self.base_model_name,
            "instruction": "!include ../prompts/analyzer.md",
            "tools": [
                "read_file",
                "search_similar_code",
                "get_code_neighbors",
                "get_code_subgraph"
            ],
            "generate_content_config": "!include ../configs/sampling.yaml"
        }
        if tool_adapter:
            analyzer_cfg["adapter"] = tool_adapter

        # Write YAML preserving !include tags
        yaml_text = (
            f"name: {analyzer_cfg['name']}\n"
            f"description: {analyzer_cfg['description']}\n"
            f"model: {analyzer_cfg['model']}\n"
            + (f"adapter: {analyzer_cfg['adapter']}\n" if "adapter" in analyzer_cfg else "")
            + "instruction: !include ../prompts/analyzer.md\n"
            + "tools:\n"
            + "  - read_file\n"
            + "  - search_similar_code\n"
            + "  - get_code_neighbors\n"
            + "  - get_code_subgraph\n"
            + "generate_content_config: !include ../configs/sampling.yaml\n"
        )
        with open(sub_agents_dir / "code_analyzer.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_text)

    def write_sampling_config(self):
        """Writes generation sampling configurations matching the official spec."""
        configs_dir = self.output_dir / "configs"
        sampling_text = (
            "temperature: 0.2\n"
            "top_p: 0.95\n"
            "max_output_tokens: 16384\n"
            "thinking_config:\n"
            "  thinking_budget: 4096\n"
            "  include_thoughts: true\n"
        )
        with open(configs_dir / "sampling.yaml", "w", encoding="utf-8") as f:
            f.write(sampling_text)

    def write_root_agent_yaml(self, main_adapter: Optional[str] = "main_lora"):
        """Writes root agent.yaml file matching official sample_submission schema."""
        agent_lines = [
            "name: swe_baseline_agent",
            f"model: {self.base_model_name}"
        ]
        if main_adapter:
            agent_lines.append(f"adapter: {main_adapter}")

        agent_lines.extend([
            "instruction: !include prompts/system.md",
            "tools:",
            "  - run_command",
            "  - read_file",
            "  - edit_file",
            "  - write_file",
            "  - get_status",
            "  - submit_patch",
            "  - get_code_neighbors",
            "  - search_similar_code",
            "  - get_code_subgraph",
            "  - agent_tool:",
            "      config_path: sub_agents/code_analyzer.yaml",
            "      skip_summarization: true",
            "generate_content_config: !include configs/sampling.yaml"
        ])

        with open(self.output_dir / "agent.yaml", "w", encoding="utf-8") as f:
            f.write("\n".join(agent_lines) + "\n")

    def write_eval_config(self):
        """Writes optional eval_config.yaml."""
        eval_cfg = (
            "evaluation:\n"
            "  timeout_seconds: 300\n"
            "  max_tool_calls: 25\n"
            "  max_time_minutes: 20\n"
            "  max_turns: 50\n"
        )
        with open(self.output_dir / "eval_config.yaml", "w", encoding="utf-8") as f:
            f.write(eval_cfg)

    def copy_adapters(self, main_adapter_dir: Optional[str] = None, tool_adapter_dir: Optional[str] = None):
        """Copies real trained PEFT LoRA adapter weights into adapters/."""
        adapters_root = self.output_dir / "adapters"
        
        # If real adapters provided, copy them
        if main_adapter_dir and Path(main_adapter_dir).exists():
            shutil.copytree(main_adapter_dir, adapters_root / "main_lora", dirs_exist_ok=True)
        if tool_adapter_dir and Path(tool_adapter_dir).exists():
            shutil.copytree(tool_adapter_dir, adapters_root / "tool_lora", dirs_exist_ok=True)

    def build(
        self,
        main_adapter_dir: Optional[str] = None,
        tool_adapter_dir: Optional[str] = None,
        adapter_path: Optional[str] = None
    ) -> Path:
        """Executes full scaffold generation."""
        if adapter_path and not main_adapter_dir:
            main_adapter_dir = adapter_path
        self.prepare_scaffold()
        self.write_prompts()
        self.write_sub_agents(tool_adapter="tool_lora" if tool_adapter_dir else None)
        self.write_sampling_config()
        self.write_root_agent_yaml(main_adapter="main_lora" if main_adapter_dir else None)
        self.write_eval_config()
        self.copy_adapters(main_adapter_dir, tool_adapter_dir)
        return self.output_dir
