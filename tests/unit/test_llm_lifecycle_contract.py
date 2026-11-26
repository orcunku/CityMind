"""Regression tests for the LLM lifecycle contract."""

import pytest

from citymind.llm.provider import LLMProvider


@pytest.mark.parametrize(
    "member",
    ["close", "closed", "__enter__", "__exit__"],
)
def test_provider_declares_lifecycle_member(member):
    assert member in LLMProvider.__dict__


class GenerateOnlyProvider:
    def generate(self, prompt, *, system_prompt=None):
        return "test response"


def test_generate_only_provider_is_incomplete():
    assert not isinstance(
        GenerateOnlyProvider(),
        LLMProvider,
    )
