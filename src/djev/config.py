from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
VLLM_VERSION = "0.30.1rc1.dev274+g5ba2e3090"
VLLM_COMMIT = "5ba2e3090b4c6bb6d5b4fd8ed3045107e787309e"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DJEV_", env_file=".env", extra="ignore")

    model: str = MODEL
    revision: str = MODEL_REVISION
    served_model: str = "diffusiongemma-jev"
    upstream: str = "http://127.0.0.1:8010"
    upstream_api_key: SecretStr | None = None
    api_key: SecretStr | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8011, ge=1, le=65535)
    canvas_length: int = Field(default=64, ge=16, le=256, multiple_of=16)
    max_model_len: int = Field(default=32768, ge=512, le=262144)
    max_num_batched_tokens: int = Field(default=16384, ge=128)
    max_concurrent_reads: int = Field(default=4, ge=1, le=128)
    max_active_requests: int = Field(default=16, ge=1, le=1024)
    timeout: float = Field(default=600, gt=0)
    startup_timeout: float = Field(default=1800, gt=0)
    max_body_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    constrained: bool = True
    gpu: str = "1"
    gpu_memory_utilization: float = Field(default=0.85, gt=0, lt=1)
    engine_python: str = ".venv-gpu/bin/python"
    runtime_cache: str = ".runtime-cache"
    hf_home: str = str(Path.home() / ".cache" / "huggingface")
