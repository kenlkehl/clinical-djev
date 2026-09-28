import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from test_core import FakeBackend

from djev.api import create_app
from djev.batch import run_batch
from djev.errors import BackendError
from djev.service import ExtractionService


def test_api_contract_and_validation_privacy(settings, compiler, request_body):
    app = create_app(settings, ExtractionService(settings, compiler, FakeBackend()))
    with TestClient(app) as client:
        result = client.post("/v1/systemone", json=request_body)
        assert result.status_code == 200
        assert set(result.json()["answers"]) == set(request_body["questions"])
        request_body["questions"]["fever"]["type"] = "PRIVATE INVALID INPUT"
        failure = client.post("/v1/systemone", json=request_body)
        assert failure.status_code == 422
        assert "PRIVATE" not in failure.text
        assert request_body["state"] not in failure.text


def test_auth_and_body_limit(settings, compiler, request_body):
    from pydantic import SecretStr

    settings.api_key = SecretStr("secret")
    settings.max_body_bytes = 1024
    app = create_app(settings, ExtractionService(settings, compiler, FakeBackend()))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 401
        assert client.post("/v1/systemone", json=request_body).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/ready", headers=headers).status_code == 200
        assert client.post("/v1/systemone", content="x" * 1025, headers=headers).status_code == 413


async def test_overload_and_slot_release(settings, compiler, request_body):
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowBackend(FakeBackend):
        async def read(self, payload, prepared):
            entered.set()
            await release.wait()
            return await super().read(payload, prepared)

    settings.max_active_requests = 1
    service = ExtractionService(settings, compiler, SlowBackend())
    app = create_app(settings, service)
    app.state.service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
        task = asyncio.create_task(c.post("/v1/systemone", json=request_body))
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert (await c.post("/v1/systemone", json=request_body)).status_code == 429
        release.set()
        assert (await task).status_code == 200
        assert (await c.post("/v1/systemone", json=request_body)).status_code == 200


def test_backend_failures_propagate(settings, compiler, request_body):
    class FailedBackend(FakeBackend):
        async def read(self, payload, prepared):
            raise BackendError("safe failure")

    app = create_app(settings, ExtractionService(settings, compiler, FailedBackend()))
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=request_body | {"samples": 3})
        assert response.status_code == 502
        assert response.json()["error"]["message"] == "safe failure"


async def test_batch_order_resume_and_input_integrity(tmp_path, settings, compiler, request_body):
    service = ExtractionService(settings, compiler, FakeBackend())
    app = create_app(settings, service)
    app.state.service = service
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(
        "".join(json.dumps(request_body | {"id": f"case-{i}"}) + "\n" for i in range(9))
    )
    transport = httpx.ASGITransport(app)
    counts = await run_batch(
        source, output, base_url="http://test", transport=transport, concurrency=2
    )
    assert counts == {"completed": 9, "failed": 0, "resumed": 0}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == [f"case-{i}" for i in range(9)]
    assert "state" not in rows[0]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows[:3]))
    counts = await run_batch(
        source, output, base_url="http://test", transport=transport, resume=True
    )
    assert counts == {"completed": 6, "failed": 0, "resumed": 3}
    with pytest.raises(FileExistsError):
        await run_batch(source, output, base_url="http://test", transport=transport)
    source.write_text(source.read_text().replace("case-0", "changed", 1))
    with pytest.raises(ValueError, match="Resume verification failed"):
        await run_batch(source, output, base_url="http://test", transport=transport, resume=True)


async def test_bad_jsonl_record_is_isolated(tmp_path, settings, compiler, request_body):
    service = ExtractionService(settings, compiler, FakeBackend())
    app = create_app(settings, service)
    app.state.service = service
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_text("BAD CLINICAL INPUT\n" + json.dumps(request_body) + "\n")
    counts = await run_batch(
        source, output, base_url="http://test", transport=httpx.ASGITransport(app)
    )
    assert counts["failed"] == counts["completed"] == 1
    assert "BAD CLINICAL INPUT" not in output.read_text()
