"""Helper for launching local offline vLLM inference engine for Gemma 4."""

from typing import List, Optional

def build_vllm_launch_command(
    model_path: str = "google/gemma-4-12b-it",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 16384,
    lora_modules: Optional[List[str]] = None
) -> List[str]:
    """Generates the shell command to launch vLLM offline server."""
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--dtype", "bfloat16"
    ]

    if lora_modules:
        cmd.append("--enable-lora")
        for lora in lora_modules:
            cmd.extend(["--lora-modules", lora])

    return cmd
