"""Live synthetic validation. Run with: uv run python scripts/smoke.py."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


async def main(url: str, output: Path):
    body = json.loads((Path(__file__).parents[1] / "examples/clinical-request.json").read_text())
    key = os.environ.get("DJEV_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(
        base_url=url, timeout=600, headers=headers, trust_env=False
    ) as client:
        (await client.get("/ready")).raise_for_status()

        async def run(name, request):
            start = time.monotonic()
            response = await client.post("/v1/systemone", json=request)
            response.raise_for_status()
            data = response.json()
            assert set(data["answers"]) == set(request["questions"])
            for question in data["diagnostics"].values():
                assert abs(sum(question["probabilities"].values()) - 1) < 1e-5
            return {"name": name, "seconds": time.monotonic() - start, "response": data}

        first = await run("all-types", body)

        def check_answers(
            result, *, fever=False, histology="adenocarcinoma", ecog=1, smoking="unknown"
        ):
            answers = result["response"]["answers"]
            return {
                "fever": (answers["fever_documented"]["noul"] > 0.5) == fever,
                "histology": answers["histology"]["choice"] == histology,
                "smoking": answers["smoking_status"]["choice"] == smoking,
                "ecog": abs(answers["ecog"]["score"] - ecog) < 0.5,
            }

        first["semantic_checks"] = check_answers(first)

        many = body | {
            "questions": {f"histology_{i}": body["questions"]["histology"] for i in range(32)}
        }
        long = body | {
            "state": (
                "SYNTHETIC HISTORICAL RECORD. No new cancer information recorded at this visit.\n"
                * 600
                + "\nLATEST VISIT:\n"
                + body["state"]
            )
        }
        results = [first, await run("question-chunks", many), await run("long-context", long)]
        results[-2]["semantic_checks"] = {
            name: answer["choice"] == "adenocarcinoma"
            for name, answer in results[-2]["response"]["answers"].items()
        }
        results[-1]["semantic_checks"] = check_answers(results[-1])
        alternate = body | {
            "state": "SYNTHETIC RECORD. Today's lung biopsy confirms squamous cell carcinoma. "
            "ECOG performance status is 3. The patient currently has fever, temperature "
            "38.6 C. The patient is a current daily cigarette smoker."
        }
        start = time.monotonic()
        parallel = await asyncio.gather(
            *(
                run(f"concurrent-{i}", (alternate if i % 2 else body) | {"seed": i})
                for i in range(4)
            )
        )
        for i, result in enumerate(parallel):
            result["semantic_checks"] = (
                check_answers(result, fever=True, histology="squamous", ecog=3, smoking="current")
                if i % 2
                else check_answers(result)
            )
        results.extend(parallel)
        concurrent_seconds = time.monotonic() - start
        # Different label sets exercise vLLM's full-vocabulary fallback when
        # constrained reads with incompatible label unions share a GPU batch.
        mixed = await asyncio.gather(
            *(
                run(f"mixed-{name}", body | {"questions": {name: question}})
                for name, question in body["questions"].items()
            )
        )
        results.extend(mixed)
        report = {"concurrent_wall_seconds": concurrent_seconds, "results": results}
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_text, json.dumps(report, indent=2) + "\n")
        failures = {
            result["name"]: [
                k for k, passed in result.get("semantic_checks", {}).items() if not passed
            ]
            for result in results
            if not all(result.get("semantic_checks", {}).values())
        }
        print(
            json.dumps(
                {
                    "report": str(output),
                    "semantic_failures": failures,
                    "runs": [
                        {"name": r["name"], "seconds": round(r["seconds"], 3)} for r in results
                    ],
                },
                indent=2,
            )
        )
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke.json"))
    args = parser.parse_args()
    asyncio.run(main(args.url, args.output))
