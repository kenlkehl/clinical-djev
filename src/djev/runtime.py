import json
import os
import subprocess
from pathlib import Path

from .config import VLLM_COMMIT, VLLM_VERSION, Settings


def engine_command(settings: Settings, port: int = 8010) -> list[str]:
    return [
        settings.engine_python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        settings.model,
        "--revision",
        settings.revision,
        "--tokenizer-revision",
        settings.revision,
        "--served-model-name",
        settings.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--safetensors-load-strategy",
        "eager",
        "--linear-backend",
        "marlin",
        "--moe-backend",
        "marlin",
        "--attention-backend",
        "TRITON_ATTN",
        "--diffusion-config",
        json.dumps({"canvas_length": settings.canvas_length}),
        "--max-model-len",
        str(settings.max_model_len),
        "--max-num-seqs",
        str(settings.max_concurrent_reads),
        "--max-num-batched-tokens",
        str(settings.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-logprobs",
        "128",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--language-model-only",
        "--enforce-eager",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]


def engine_environment(settings: Settings) -> dict[str, str]:
    env = os.environ.copy()
    cache = Path(settings.runtime_cache).resolve()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": settings.gpu,
            "HF_HOME": settings.hf_home,
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    if settings.upstream_api_key:
        env["VLLM_API_KEY"] = settings.upstream_api_key.get_secret_value()
    return env


def inspect_runtime(settings: Settings) -> dict:
    """Inspect runtime source as well as its version: old vLLM ignores xargs."""
    code = """
import importlib.metadata, json, pathlib
dist = importlib.metadata.distribution("vllm")
source = pathlib.Path(dist.locate_file("vllm/model_executor/models/diffusion_gemma.py")).read_text()
import torch
print(json.dumps({
    "version": dist.version,
    "structured_reads": all(x in source for x in [
        "diffusion_seed_canvas", "diffusion_read_only", "diffusion_constrained"]),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}))
"""
    result = subprocess.run(
        [settings.engine_python, "-c", code],
        env=engine_environment(settings),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def install_runtime(directory: Path, python: str):
    directory.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(directory.parent / ".uv-cache")
    runtime = directory
    if not (runtime / "bin/python").exists():
        subprocess.run(["uv", "venv", "--python", python, str(runtime)], env=env, check=True)
    lock = Path(__file__).with_name("runtime.lock")
    requirements = ["--requirements", str(lock)] if lock.exists() else [f"vllm=={VLLM_VERSION}"]
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(runtime / "bin/python"),
            *requirements,
            "--extra-index-url",
            f"https://wheels.vllm.ai/{VLLM_COMMIT}",
            "--index-strategy",
            "unsafe-best-match",
        ],
        env=env,
        check=True,
    )
