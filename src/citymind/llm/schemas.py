"""Structured output schemas for CITYMIND LLM analysis."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


AnalysisText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]

EvidenceText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class InvestigationOutput(BaseModel):
    """Advisory LLM output without action authority."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    summary: AnalysisText

    evidence: tuple[EvidenceText, ...] = Field(
        min_length=1,
        max_length=10,
    )

    uncertainties: tuple[EvidenceText, ...] = Field(
        max_length=10,
    )

    recommendation: Literal[
        "no_additional_review",
        "continue_monitoring",
        "request_additional_data",
        "request_human_review",
    ]


    @field_validator("evidence", "uncertainties", mode="before")
    @classmethod
    def freeze_sequences(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value
