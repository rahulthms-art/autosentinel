import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar.db import DB  # noqa: E402
from radar.llm import Router  # noqa: E402


class FakeLLM:
    """Stands in for the OpenAI client; answers by matching the system prompt."""

    def __init__(self, rules):
        self.rules = rules  # list of (substring_in_system_prompt, reply or callable)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, **kw):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append((model, system[:40]))
        for needle, reply in self.rules:
            if needle in system:
                text = reply(user) if callable(reply) else reply
                break
        else:
            text = "{}"
        return NS(choices=[NS(message=NS(content=text))],
                  usage=NS(prompt_tokens=100, completion_tokens=50))


@pytest.fixture
def db():
    return DB(":memory:")


@pytest.fixture
def make_router():
    def _make(rules):
        fake = FakeLLM(rules)
        return Router(fake, "nvidia/nemotron-nano", "nvidia/nemotron-ultra"), fake
    return _make
