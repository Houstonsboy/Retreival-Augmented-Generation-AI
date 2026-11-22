"""Pydantic models for the Contract Evidence Analyst output.

Mirrors the ``# OUTPUT`` schema in ``system_prompt.md`` exactly, so the parsed
LLM response can be validated, serialised and verified by the deterministic
rules layer. Unknown keys are ignored and missing keys become empty so a
partially-formed response is still usable.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntegrityRating(str, Enum):
    STRONG = "STRONG"
    ADEQUATE = "ADEQUATE"
    WEAK = "WEAK"


class ElementStatus(str, Enum):
    PRESENT = "PRESENT"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"
    UNCLEAR = "UNCLEAR"
    NO_ISSUE_INDICATED = "NO_ISSUE_INDICATED"
    NOT_REQUIRED = "NOT_REQUIRED"


class PaymentClassification(str, Enum):
    VERIFIED_LOOKING = "VERIFIED-LOOKING"
    PRESENT_BUT_UNVERIFIED = "PRESENT BUT UNVERIFIED"
    SUSPICIOUS = "SUSPICIOUS"
    CLAIMED_ONLY = "CLAIMED ONLY"
    ABSENT = "ABSENT"


class OverallVerdict(str, Enum):
    LIKELY_VALID = "LIKELY VALID"
    POSSIBLY_VALID_WITH_GAPS = "POSSIBLY VALID WITH GAPS"
    LIKELY_NOT_VALID = "LIKELY NOT VALID"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


ELEMENT_KEYS = [chr(c) for c in range(ord("A"), ord("K") + 1)]  # A..K


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    msg_index: int | None = None
    sender: str | None = None
    time: str | None = None
    quote: str = ""


class Party(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    role: str | None = None
    notes: str | None = None


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    msg_index: int | None = None
    evidence_id: str | None = None
    sender: str | None = None
    time: str | None = None
    event: str = ""


class Element(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: ElementStatus
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float | None = None


class MpesaPayment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    amount_kes: float | None = None
    sender: str | None = None
    recipient: str | None = None
    datetime: str | None = None
    evidence_id: str | None = None
    basis: list[str] = Field(default_factory=list)
    classification: PaymentClassification
    notes: str = ""


class PartyQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    element: str = ""
    question: str = ""


class FollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    element: str = ""
    type: str = ""
    prompt: str = ""
    why_it_matters: str = ""
    blocking: bool = False
    target: str | None = None
    options: list[str] = Field(default_factory=list)
    date_role: str | None = None


class UnrelatedMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    msg_index: int | None = None
    text: str = ""


class Integrity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rating: IntegrityRating
    issues: list[str] = Field(default_factory=list)


class AnalystResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_integrity: Integrity
    unrelated_messages: list[UnrelatedMessage] = Field(default_factory=list)
    parties: list[Party] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    elements: dict[str, Element] = Field(default_factory=dict)
    mpesa_payments: list[MpesaPayment] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    overall_verdict: OverallVerdict
    confidence: float | None = None
    changes_since_previous: list[str] = Field(default_factory=list)
    plain_summary: str = ""
    strengthen_checklist: list[str] = Field(default_factory=list)
    follow_up_requests: list[FollowUpRequest] = Field(default_factory=list)
    questions_for_parties: list[PartyQuestion] = Field(default_factory=list)
    disclaimer: str = ""

    @field_validator("elements", mode="before")
    @classmethod
    def _normalize_elements(cls, value):
        if not isinstance(value, dict):
            return value
        out: dict[str, object] = {}
        for key, val in value.items():
            upper = str(key).strip().upper()
            if upper in ELEMENT_KEYS and isinstance(val, dict):
                out[upper] = val
        return out
