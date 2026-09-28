import asyncio
import math
import secrets
import time
from statistics import mean, stdev

from .backend import Backend, ReadResult
from .compiler import Compiler, PreparedRead, Slot
from .config import Settings
from .errors import InvalidRequest
from .schemas import (
    ChoiceAnswer,
    ExtractionRequest,
    ExtractionResponse,
    NoulAnswer,
    QuestionDiagnostics,
    ScoreAnswer,
)


def entropy(probabilities) -> float:
    value = -sum(p * math.log(p) for p in probabilities if p > 0) / math.log(len(probabilities))
    return min(1.0, max(0.0, value))


def stderr(values: list[float]) -> float | None:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def summarize(slot: Slot, samples: list[tuple[float, ...]], masses: list[float]):
    average = [mean(row[i] for row in samples) for i in range(len(slot.names))]
    winner = max(range(len(average)), key=average.__getitem__)
    keys = [str(i) for i in range(len(average))] if slot.question.type == "score" else slot.names
    probabilities = dict(zip(keys, average, strict=True))
    if slot.question.type == "noul":
        answer = NoulAnswer(noul=average[0])
        values = [row[0] for row in samples]
    elif slot.question.type == "choice":
        answer = ChoiceAnswer(
            choice=slot.names[winner], probabilities=probabilities, confidence=average[winner]
        )
        values = [row[winner] for row in samples]
    else:
        answer = ScoreAnswer(
            score=sum(i * p for i, p in enumerate(average)),
            legend=dict(zip(keys, slot.names, strict=True)),
            probabilities=probabilities,
            confidence=average[winner],
        )
        values = [sum(i * p for i, p in enumerate(row)) for row in samples]
    diagnostics = QuestionDiagnostics(
        samples=len(samples),
        probabilities=probabilities,
        normalized_entropy=entropy(average),
        probability_stderr=(
            {key: stderr([row[i] for row in samples]) for i, key in enumerate(keys)}
            if len(samples) > 1
            else None
        ),
        value_stderr=stderr(values),
        agreement=mean(max(range(len(row)), key=row.__getitem__) == winner for row in samples),
        label_mass=mean(masses),
    )
    return answer, diagnostics


class ExtractionService:
    def __init__(self, settings: Settings, compiler: Compiler, backend: Backend):
        self.settings, self.compiler, self.backend = settings, compiler, backend

    async def _group(self, prepared: PreparedRead, request: ExtractionRequest, seed: int):
        async def one(index: int):
            payload = self.compiler.payload(
                prepared,
                model=self.settings.served_model,
                seed=seed + index * 7919,
                steps=request.steps,
                constrained=self.settings.constrained,
            )
            return await self.backend.read(payload, prepared)

        async def many(indices):
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one(i)) for i in indices]
            return [task.result() for task in tasks]

        if request.samples == "auto":
            results = [await one(0)]
            if (
                request.auto_max > 1
                and max(map(entropy, results[0].probabilities)) > request.auto_threshold
            ):
                results += await many(range(1, request.auto_max))
            return results
        return await many(range(request.samples))

    async def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        if request.model not in (None, self.settings.served_model, self.settings.model):
            raise InvalidRequest("Requested model does not match the configured backend")
        started = time.monotonic()
        # CPU tokenization must not block health checks or other requests.
        prepared = await asyncio.to_thread(self.compiler.prepare, request)
        seed = request.seed if request.seed is not None else secrets.randbits(62)
        answers, diagnostics = {}, {}
        usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        reads = 0
        # Keep each clinical document's groups sequential: this limits prompt copies
        # and allows vLLM prefix caching to reuse the completed prefill between reads.
        for i, item in enumerate(prepared):
            try:
                results: list[ReadResult] = await self._group(item, request, seed + i * 104729)
            except ExceptionGroup as exc:
                error = exc
                while isinstance(error, BaseExceptionGroup):
                    error = error.exceptions[0]
                raise error from None
            reads += len(results)
            for result in results:
                for key in usage:
                    usage[key] += result.usage[key]
            for j, slot in enumerate(item.group.slots):
                answers[slot.key], diagnostics[slot.key] = summarize(
                    slot, [r.probabilities[j] for r in results], [r.label_mass[j] for r in results]
                )
        return ExtractionResponse(
            model=self.settings.served_model,
            answers=answers,
            diagnostics=diagnostics,
            usage=usage,
            metadata={
                "seed": seed,
                "groups": len(prepared),
                "reads": reads,
                "steps": request.steps,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                "max_prompt_tokens": max(len(p.prompt) for p in prepared),
                "vllm_version": self.backend.version,
                "model_revision": self.settings.revision,
                "constrained": self.settings.constrained,
                "probability_semantics": "label-conditional; not clinically calibrated",
            },
        )
