from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16384)]
# Preserve caller IDs/options exactly: stripping map keys can silently merge them.
Name = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Text
    criteria: dict[Literal["true", "false"], Text | None] | None = None


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Text
    criteria: dict[Name, Text | None] = Field(min_length=2, max_length=26)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Text
    criteria: list[Text] = Field(min_length=2, max_length=26)

    @model_validator(mode="after")
    def distinct_levels(self):
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("score levels must be distinct")
        return self


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class ExtractionRequest(StrictModel):
    model: str | None = None
    state: JsonValue
    questions: dict[Name, Question] = Field(min_length=1, max_length=512)
    instructions: Text | None = None
    samples: Annotated[int, Field(ge=1, le=32)] | Literal["auto"] = 1
    auto_max: int = Field(default=4, ge=1, le=32)
    auto_threshold: float = Field(default=0.25, ge=0, le=1)
    steps: int = Field(default=1, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def nonempty_state(self):
        if self.state is None or (isinstance(self.state, str) and not self.state.strip()):
            raise ValueError("state must contain clinical text or a JSON value")
        return self


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class QuestionDiagnostics(StrictModel):
    samples: int
    probabilities: dict[str, float]
    normalized_entropy: float
    probability_stderr: dict[str, float] | None
    value_stderr: float | None
    agreement: float
    label_mass: float


class ExtractionResponse(StrictModel):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    diagnostics: dict[str, QuestionDiagnostics]
    usage: dict[str, int]
    metadata: dict[str, JsonValue]
