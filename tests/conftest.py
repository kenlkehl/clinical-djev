import re

import pytest

from djev.compiler import Compiler
from djev.config import Settings
from djev.schemas import ExtractionRequest


class Tokenizer:
    vocab_size = 262144
    pad_token_id = 0

    def __init__(self):
        self.vocabulary = {"<turn|>": 106, "<|channel>": 100, "<channel|>": 101}
        self.last_messages = None

    def encode(self, text, **kwargs):
        tokens = re.findall(r"<[^>]+>|\w+|[^\w]", text)
        for token in tokens:
            self.vocabulary.setdefault(token, 1000 + len(self.vocabulary))
        return [self.vocabulary[t] for t in tokens]

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return self.encode("\n".join(m["content"] for m in messages))


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture
def compiler(settings):
    return Compiler(Tokenizer(), settings.canvas_length, settings.max_model_len)


@pytest.fixture
def request_body():
    return {
        "state": "Synthetic: fever denied; histology is adenocarcinoma; ECOG 1.",
        "questions": {
            "fever": {"type": "noul", "instructions": "Is fever present?"},
            "histology": {
                "type": "choice",
                "instructions": "Histology?",
                "criteria": {"adeno": None, "squamous": None},
            },
            "ecog": {
                "type": "score",
                "instructions": "ECOG?",
                "criteria": ["0", "1", "2"],
            },
        },
        "seed": 42,
    }


@pytest.fixture
def extraction(request_body):
    return ExtractionRequest.model_validate(request_body)
