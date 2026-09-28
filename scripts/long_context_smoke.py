"""Live long-context capacity check with independent synthetic records."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from djev.compiler import Compiler
from djev.config import Settings
from djev.schemas import ExtractionRequest

FILLER = "SYNTHETIC HISTORICAL RECORD. No new cancer information recorded at this visit.\n"


def prepare_cases(target: int, settings: Settings):
    body = json.loads((Path(__file__).parents[1] / "examples/clinical-request.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model,
        revision=settings.revision,
        cache_dir=str(Path(settings.hf_home) / "hub"),
        local_files_only=True,
        trust_remote_code=False,
    )
    compiler = Compiler(tokenizer, settings.canvas_length, 262144)
    cases = []
    for name, positive in (("single", False), ("parallel-a", False), ("parallel-b", True)):
        # Distinct first state lines prevent reuse of the long document's KV
        # prefix between cases. Evidence is spread across the document.
        facts = (
            [
                "Lung biopsy: squamous cell carcinoma.",
                "Current fever is present, 38.6 C.",
                "ECOG performance status 3.",
                "Current daily cigarette smoker.",
            ]
            if positive
            else [
                "Lung biopsy: adenocarcinoma.",
                "Current fever and chills are explicitly denied.",
                "ECOG performance status 1.",
                "Smoking history is not documented.",
            ]
        )
        repeats = target // 17
        while True:
            state = f"SYNTHETIC PATIENT {name}. All clinical facts concern the current visit.\n"
            for i, fact in enumerate(facts):
                state += fact + "\n"
                if i < 3:
                    state += FILLER * (repeats // 3 + (i < repeats % 3))
            request = body | {"state": state}
            tokens = len(compiler.prepare(ExtractionRequest.model_validate(request))[0].prompt)
            if tokens >= target:
                break
            repeats += max(1, math.ceil((target - tokens) / 17))
        cases.append((name, positive, request, tokens))
    return cases


async def run(url: str, output: Path, cases, settings: Settings):
    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"
    async with httpx.AsyncClient(
        base_url=url, timeout=settings.timeout, headers=headers, trust_env=False
    ) as client:
        (await client.get("/ready")).raise_for_status()

        async def one(case):
            name, positive, request, tokens = case
            started = time.monotonic()
            response = await client.post("/v1/systemone", json=request)
            response.raise_for_status()
            data = response.json()
            assert data["metadata"]["max_prompt_tokens"] == tokens
            answers = data["answers"]
            checks = {
                "fever": (answers["fever_documented"]["noul"] > 0.5) == positive,
                "histology": answers["histology"]["choice"]
                == ("squamous" if positive else "adenocarcinoma"),
                "ecog": abs(answers["ecog"]["score"] - (3 if positive else 1)) < 0.5,
                "smoking": answers["smoking_status"]["choice"]
                == ("current" if positive else "unknown"),
            }
            result = {
                "name": name,
                "tokens": tokens,
                "seconds": time.monotonic() - started,
                "semantic_checks": checks,
                "response": data,
            }
            print(json.dumps({k: v for k, v in result.items() if k != "response"}), flush=True)
            return result

        results = [await one(cases[0])]
        started = time.monotonic()
        results.extend(await asyncio.gather(*(one(case) for case in cases[1:])))
        report = {"parallel_wall_seconds": time.monotonic() - started, "results": results}
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_text, json.dumps(report, indent=2) + "\n")
        print(f"Report: {output}", flush=True)
        if not all(all(r["semantic_checks"].values()) for r in results):
            raise SystemExit("Capacity test ran, but some synthetic answer checks failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--tokens", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("outputs/long-context-smoke.json"))
    args = parser.parse_args()
    if not 1000 <= args.tokens <= 260000:
        parser.error("--tokens must be between 1,000 and 260,000")
    settings = Settings()
    asyncio.run(run(args.url, args.output, prepare_cases(args.tokens, settings), settings))
