"""Bounded-memory, ordered JSONL processing with verified resume."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from pydantic import ValidationError

from .schemas import ExtractionRequest, ExtractionResponse


def fingerprint(line: str) -> str:
    return hashlib.sha256(line.rstrip("\r\n").encode()).hexdigest()


async def run_batch(
    source: Path,
    destination: Path,
    *,
    base_url: str,
    concurrency: int = 4,
    api_key: str | None = None,
    resume: bool = False,
    timeout: float = 600,
    transport=None,
) -> dict[str, int]:
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    if concurrency < 1 or concurrency > 128:
        raise ValueError("concurrency must be between 1 and 128")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    counts = {"completed": 0, "failed": 0, "resumed": 0}
    with source.open() as input_file:
        line_number = 0
        if resume and destination.exists():
            with destination.open() as previous:
                for old in previous:
                    line = input_file.readline()
                    line_number += 1
                    try:
                        record = json.loads(old)
                        matches = (
                            line
                            and old.endswith("\n")
                            and record["line"] == line_number
                            and record["request_sha256"] == fingerprint(line)
                            and ("result" in record or "error" in record)
                        )
                    except (ValueError, KeyError, TypeError):
                        matches = False
                    if not matches:
                        raise ValueError(f"Resume verification failed at output line {line_number}")
                    counts["resumed"] += 1
        mode = "a" if resume else "x"
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            trust_env=False,
            transport=transport,
        ) as client:

            async def process(line: str, number: int):
                record = {"line": number, "request_sha256": fingerprint(line)}
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError()
                    record["id"] = row.pop("id", str(number))
                    request = ExtractionRequest.model_validate(row)
                except (ValueError, TypeError, ValidationError):
                    return dict(record, error={"code": "invalid_request"})
                # Retry only overload and transport failures, using the same seed.
                body = request.model_dump(mode="json")
                if body["seed"] is None:
                    body["seed"] = int(record["request_sha256"][:15], 16)
                for attempt in range(4):
                    try:
                        response = await client.post("/v1/systemone", json=body)
                        if response.status_code in (429, 502, 503, 504) and attempt < 3:
                            await asyncio.sleep(2**attempt)
                            continue
                        if response.is_error:
                            return dict(
                                record,
                                error={
                                    "code": "http_error",
                                    "status": response.status_code,
                                },
                            )
                        result = ExtractionResponse.model_validate(response.json())
                        return dict(record, result=result.model_dump(mode="json"))
                    except httpx.TransportError:
                        if attempt < 3:
                            await asyncio.sleep(2**attempt)
                            continue
                        return dict(record, error={"code": "transport_error"})
                    except (ValueError, ValidationError):
                        return dict(record, error={"code": "invalid_response"})

            # A bounded window preserves input order without collecting the corpus.
            with destination.open(mode) as output_file:
                while True:
                    lines = []
                    for _ in range(concurrency):
                        line = input_file.readline()
                        if not line:
                            break
                        line_number += 1
                        lines.append((line, line_number))
                    if not lines:
                        break
                    async with asyncio.TaskGroup() as group:
                        tasks = [group.create_task(process(line, n)) for line, n in lines]
                    for task in tasks:
                        result = task.result()
                        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_file.flush()
                        counts["failed" if "error" in result else "completed"] += 1
    return counts
