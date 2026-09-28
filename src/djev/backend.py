import asyncio
import math
from dataclasses import dataclass

import httpx
from packaging.version import InvalidVersion, Version

from .compiler import PreparedRead
from .config import VLLM_VERSION, Settings
from .errors import BackendError, BackendUnavailable


@dataclass(frozen=True)
class ReadResult:
    probabilities: tuple[tuple[float, ...], ...]
    label_mass: tuple[float, ...]
    usage: dict[str, int]


def parse_result(data: dict, read: PreparedRead) -> ReadResult:
    """Require every requested label. Missing probabilities are never imputed."""
    try:
        rows = data["choices"][0]["logprobs"]["top_logprobs"]
        distributions, masses = [], []
        for slot in read.group.slots:
            row = rows[slot.position]
            values = [float(row[f"token_id:{token}"]) for token in slot.token_ids]
            if not all(math.isfinite(v) and v <= 0.001 for v in values):
                raise ValueError("invalid logprob")
            maximum = max(values)
            weights = [math.exp(value - maximum) for value in values]
            total = sum(weights)
            distributions.append(tuple(w / total for w in weights))
            masses.append(min(1.0, sum(math.exp(v) for v in values)))
        usage = {
            key: int(data.get("usage", {}).get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return ReadResult(tuple(distributions), tuple(masses), usage)
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        raise BackendError(
            "vLLM did not return finite exact logprobs at every answer slot. "
            "Check the pinned nightly, tokenizer revision, and canvas configuration."
        ) from None


class Backend:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        headers = {}
        if settings.upstream_api_key:
            headers["Authorization"] = f"Bearer {settings.upstream_api_key.get_secret_value()}"
        self.client = client or httpx.AsyncClient(
            base_url=settings.upstream.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout, connect=10),
            limits=httpx.Limits(max_connections=settings.max_concurrent_reads + 2),
            trust_env=False,
        )
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_reads)
        self.version: str | None = None

    async def close(self):
        await self.client.aclose()

    async def check(self) -> int:
        """Check model identity, context limit, and minimum protocol version."""
        try:
            version_response = await self.client.get("/version")
            version_response.raise_for_status()
            version = version_response.json()["version"]
            if Version(version) < Version(VLLM_VERSION):
                raise BackendUnavailable(f"vLLM {VLLM_VERSION} or newer is required")
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            model = next(
                m for m in response.json()["data"] if m["id"] == self.settings.served_model
            )
            if model.get("root") != self.settings.model:
                raise BackendUnavailable("The backend model root does not match DJEV_MODEL")
            self.version = version
            return min(self.settings.max_model_len, int(model["max_model_len"]))
        except BackendUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration, InvalidVersion):
            raise BackendUnavailable(
                "Cannot verify the vLLM backend, served model, or context limit"
            ) from None

    async def read(self, payload: dict, prepared: PreparedRead) -> ReadResult:
        try:
            async with self.semaphore:
                response = await self.client.post("/v1/completions", json=payload)
            if response.status_code in (429, 503):
                raise BackendUnavailable("vLLM is temporarily unavailable")
            if response.is_error:
                # Upstream errors can echo the prompt. Never relay or log them.
                raise BackendError(f"vLLM rejected the read (HTTP {response.status_code})")
            return parse_result(response.json(), prepared)
        except httpx.TimeoutException:
            raise BackendUnavailable("vLLM read timed out") from None
        except httpx.HTTPError:
            raise BackendUnavailable("Could not reach vLLM") from None
        except ValueError:
            raise BackendError("vLLM returned invalid JSON") from None
