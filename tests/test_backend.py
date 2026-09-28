import asyncio
import math

import httpx
import pytest

from djev.backend import Backend
from djev.config import VLLM_VERSION
from djev.errors import BackendError, BackendUnavailable


@pytest.mark.parametrize("version, accepted", [("0.30.0", False), (VLLM_VERSION, True)])
async def test_protocol_version_guard(settings, version, accepted):
    def handle(request):
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": version})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": settings.served_model,
                        "root": settings.model,
                        "max_model_len": 16384,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://test",
    ) as client:
        backend = Backend(settings, client)
        if accepted:
            assert await backend.check() == 16384
        else:
            with pytest.raises(BackendUnavailable, match="required"):
                await backend.check()


async def test_read_concurrency_cap(settings, compiler, extraction):
    settings.max_concurrent_reads = 2
    active, peak = 0, 0
    prepared = compiler.prepare(extraction)[0]
    rows = [{} for _ in prepared.group.template]
    for slot in prepared.group.slots:
        rows[slot.position] = {
            f"token_id:{token}": math.log(1 / len(slot.token_ids)) for token in slot.token_ids
        }

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": rows}}]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://test",
    ) as client:
        backend = Backend(settings, client)
        await asyncio.gather(*(backend.read({}, prepared) for _ in range(8)))
        assert peak == 2


async def test_upstream_body_is_not_exposed(settings, compiler, extraction):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text="PATIENT DATA")),
        base_url="http://test",
    ) as client:
        with pytest.raises(BackendError) as exc:
            await Backend(settings, client).read({}, compiler.prepare(extraction)[0])
        assert "PATIENT" not in str(exc.value)
