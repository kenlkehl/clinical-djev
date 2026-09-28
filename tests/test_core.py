import math

import pytest
from pydantic import ValidationError

from djev.backend import ReadResult, parse_result
from djev.errors import BackendError, ContextTooLong
from djev.schemas import ExtractionRequest
from djev.service import ExtractionService, summarize


@pytest.mark.parametrize(
    "patch",
    [
        {"samples": True},
        {"samples": 0},
        {"samples": 33},
        {"steps": 0},
        {"auto_threshold": float("nan")},
        {"state": " "},
        {"state": None},
        {"questions": {}},
        {"unsupported": True},
    ],
)
def test_invalid_request(request_body, patch):
    with pytest.raises(ValidationError):
        ExtractionRequest.model_validate(request_body | patch)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "x"}},
        {"type": "choice", "instructions": "x", "criteria": {"a": None}},
        {"type": "score", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": list(map(str, range(27)))},
    ],
)
def test_invalid_criteria(question):
    with pytest.raises(ValidationError):
        ExtractionRequest(state="x", questions={"q": question})


def test_mapping_keys_are_not_silently_merged(extraction):
    q = extraction.questions["fever"]
    request = ExtractionRequest(state="x", questions={"a": q, " a": q})
    assert list(request.questions) == ["a", " a"]


def test_seeded_canvas_and_pins(compiler, extraction):
    prepared = compiler.prepare(extraction)[0]
    payload = compiler.payload(prepared, model="m", seed=1, steps=2, constrained=True)
    xargs = payload["vllm_xargs"]
    assert len(xargs["diffusion_seed_canvas"]) == xargs["diffusion_canvas_length"]
    free = {s.position for s in prepared.group.slots}
    assert set(xargs["diffusion_pinned"]) == set(range(prepared.group.width)) - free
    assert xargs["diffusion_read_only"]
    assert payload["logprob_token_ids"] == prepared.group.label_ids
    assert "seed" not in payload and "temperature" not in payload
    assert payload == compiler.payload(prepared, model="m", seed=1, steps=2, constrained=True)
    for p in range(len(prepared.group.template)):
        if p not in free:
            assert xargs["diffusion_seed_canvas"][p] == prepared.group.template[p]


def test_question_chunking_keeps_full_state_and_shared_prompt(compiler, extraction):
    extraction.questions = {f"external ID {i}": extraction.questions["fever"] for i in range(40)}
    prepared = compiler.prepare(extraction)
    assert len(prepared) > 1
    assert all(p.group.width <= compiler.canvas_length for p in prepared)
    assert len({p.prompt for p in prepared}) == 1
    assert compiler.tokenizer.last_messages[1]["content"] == extraction.state
    keys = [slot.key for p in prepared for slot in p.group.slots]
    assert keys == list(extraction.questions)
    assert "Question q39:" in prepared[0].group.system


def test_no_truncation(compiler, extraction):
    compiler.max_model_len = 20
    with pytest.raises(ContextTooLong, match="No text was truncated"):
        compiler.prepare(extraction)


def test_exact_logprobs_required(compiler, extraction):
    prepared = compiler.prepare(extraction)[0]
    rows = [{} for _ in prepared.group.template]
    for slot in prepared.group.slots:
        rows[slot.position] = {
            f"token_id:{token}": math.log(1 / len(slot.token_ids)) for token in slot.token_ids
        }
    data = {"choices": [{"logprobs": {"top_logprobs": rows}}]}
    result = parse_result(data, prepared)
    assert all(sum(p) == pytest.approx(1) for p in result.probabilities)
    rows[prepared.group.slots[0].position].popitem()
    with pytest.raises(BackendError, match="exact logprobs"):
        parse_result(data, prepared)


def test_score_is_zero_based_expected_value(compiler, extraction):
    slot = compiler.prepare(extraction)[0].group.slots[2]
    answer, diagnostics = summarize(slot, [(0.1, 0.2, 0.7)], [0.8])
    assert answer.score == pytest.approx(1.6)
    assert answer.legend == {"0": "0", "1": "1", "2": "2"}
    assert answer.confidence == 0.7
    assert diagnostics.value_stderr is None


def test_noul_is_yes_probability_and_correct_stderr(compiler, extraction):
    slot = compiler.prepare(extraction)[0].group.slots[0]
    answer, diagnostics = summarize(slot, [(0.1, 0.9), (0.3, 0.7)], [1, 1])
    assert answer.noul == pytest.approx(0.2)
    assert diagnostics.value_stderr == pytest.approx(0.1)
    assert diagnostics.agreement == 1


class FakeBackend:
    version = "test"

    def __init__(self, probability=0.5):
        self.calls = []
        self.probability = probability

    async def check(self):
        return 32768

    async def read(self, payload, prepared):
        self.calls.append(payload)
        rows = tuple(
            (self.probability, 1 - self.probability)
            if len(s.names) == 2
            else tuple(1 / len(s.names) for _ in s.names)
            for s in prepared.group.slots
        )
        return ReadResult(
            rows,
            tuple(1.0 for _ in rows),
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        )


@pytest.mark.parametrize("probability, count", [(0.5, 4), (0.9999, 1)])
async def test_adaptive_sampling(compiler, settings, extraction, probability, count):
    extraction.questions = {"fever": extraction.questions["fever"]}
    extraction.samples = "auto"
    backend = FakeBackend(probability)
    result = await ExtractionService(settings, compiler, backend).extract(extraction)
    assert len(backend.calls) == count
    assert result.metadata["reads"] == count
    assert result.usage["prompt_tokens"] == count * 100
    assert len({tuple(p["vllm_xargs"]["diffusion_seed_canvas"]) for p in backend.calls}) == count


async def test_fixed_sampling(compiler, settings, extraction):
    extraction.samples = 3
    result = await ExtractionService(settings, compiler, FakeBackend()).extract(extraction)
    assert set(result.answers) == set(extraction.questions)
    assert result.metadata["reads"] == 3
