import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import uvicorn

from .batch import run_batch
from .config import Settings
from .runtime import engine_command, engine_environment, inspect_runtime, install_runtime


def serve(settings: Settings, *, launch: bool):
    from .api import create_app

    process = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        if launch:
            info = inspect_runtime(settings)
            if not info["structured_reads"] or not info["cuda_available"]:
                raise RuntimeError(f"GPU runtime is not ready: {info}")
            url = httpx.URL(settings.upstream)
            if url.host != "127.0.0.1" or url.scheme != "http" or url.path not in ("", "/"):
                raise ValueError("Managed serve requires an http://127.0.0.1:PORT upstream")
            with httpx.Client(timeout=2, trust_env=False) as client:
                try:
                    client.get(f"{settings.upstream}/health")
                except httpx.TransportError:
                    pass
                else:
                    raise RuntimeError("Upstream port is already occupied; use `djev api`")
            process = subprocess.Popen(
                engine_command(settings, url.port or 80),
                env=engine_environment(settings),
                start_new_session=True,
            )
            deadline = time.monotonic() + settings.startup_timeout
            headers = {}
            if settings.upstream_api_key:
                headers["Authorization"] = f"Bearer {settings.upstream_api_key.get_secret_value()}"
            with httpx.Client(timeout=2, headers=headers, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited with code {process.returncode}")
                    try:
                        if client.get(f"{settings.upstream}/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError("vLLM startup timed out")
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description="clinical-djev structured extraction")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "api", "engine-command", "doctor"):
        command = sub.add_parser(name)
        for flag in ("gpu", "engine-python", "upstream", "host"):
            command.add_argument("--" + flag)
        command.add_argument("--port", type=int)
        command.add_argument("--max-model-len", type=int)
        command.add_argument("--max-num-batched-tokens", type=int)
    install = sub.add_parser("install-runtime")
    install.add_argument("--directory", type=Path, default=Path(".venv-gpu"))
    install.add_argument("--python", default="3.12")
    sub.add_parser("download-model")
    batch = sub.add_parser("batch")
    batch.add_argument("input", type=Path)
    batch.add_argument("output", type=Path)
    batch.add_argument("--url", default="http://127.0.0.1:8011")
    batch.add_argument("--concurrency", type=int, default=4)
    batch.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    overrides = {
        k: v for k, v in vars(args).items() if k in Settings.model_fields and v is not None
    }
    settings = Settings(**overrides)
    try:
        if args.command == "install-runtime":
            install_runtime(args.directory, args.python)
        elif args.command == "download-model":
            os.environ["HF_HOME"] = settings.hf_home
            # Sequential HTTP writes avoid random-write stalls on SSHFS/NFS model stores.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from huggingface_hub import snapshot_download

            print(
                snapshot_download(
                    settings.model,
                    revision=settings.revision,
                    max_workers=1,
                    cache_dir=str(Path(settings.hf_home) / "hub"),
                )
            )
        elif args.command == "engine-command":
            env = engine_environment(settings)
            prefix = " ".join(
                f"{key}={shlex.quote(env[key])}"
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "HF_HOME",
                    "VLLM_USE_V2_MODEL_RUNNER",
                )
            )
            print(
                prefix
                + " "
                + shlex.join(engine_command(settings, httpx.URL(settings.upstream).port or 80))
            )
        elif args.command == "doctor":
            print(json.dumps(inspect_runtime(settings), indent=2))
        elif args.command == "batch":
            counts = asyncio.run(
                run_batch(
                    args.input,
                    args.output,
                    base_url=args.url,
                    concurrency=args.concurrency,
                    resume=args.resume,
                    timeout=settings.timeout,
                    api_key=settings.api_key.get_secret_value() if settings.api_key else None,
                )
            )
            print(json.dumps(counts))
            if counts["failed"]:
                raise SystemExit(1)
        else:
            serve(settings, launch=args.command == "serve")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"djev: {exc}\n")
    except KeyboardInterrupt:
        sys.exit(130)
