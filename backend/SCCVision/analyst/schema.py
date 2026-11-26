"""Pydantic models for the new ContractFacts LLM output schema (Phase 4b).

The LLM extracts FACTS with msg_index citations and advisory opinions.
All statuses, verdicts, confidence and eligibility are derived by code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Party = Literal["PARTY_1", "PARTY_2", "UNCLEAR"]
Status = Literal[
    "PRESENT", "PARTIAL", "ABSENT", "UNCLEAR",
    "NO_ISSUE_INDICATED", "NOT_REQUIRED",
]


class Ref(BaseModel):
    model_config = ConfigDict(extra="ignore")

    msg_index: int
    span: str | None = None


class PaymentFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ref: Ref | None = None
    mpesa_code: str | None = None
    amount: int | None = None
    payer: Party = "UNCLEAR"
    classification: Literal[
        "VERIFIED-LOOKING", "PRESENT BUT UNVERIFIED", "SUSPICIOUS", "CLAIMED ONLY"
    ] = "CLAIMED ONLY"


class IntegrityConcern(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Literal[
        "EDIT_INDICATOR", "DELETION_INDICATOR", "INCONSISTENT_ACCOUNT",
        "MISSING_CONTEXT", "POSSIBLE_FABRICATION", "THREAT_OR_COERCION",
    ]
    refs: list[Ref] = Field(default_factory=list)


class ElementOpinion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Status
    reason: str = ""


class Narrative(BaseModel):
    model_config = ConfigDict(extra="ignore")

    facts_summary: str = ""
    red_flag_notes: list[str] = Field(default_factory=list)
    questions_for_parties: list[str] = Field(default_factory=list)
    strengthen_suggestions: list[str] = Field(default_factory=list)


class ContractFacts(BaseModel):
    """Full structured output from call 1 (inspect). Code derives all statuses from this."""
    model_config = ConfigDict(extra="ignore")

    claimant: Party = "UNCLEAR"
    respondent: Party = "UNCLEAR"
    party_identity_refs: dict[str, list[Ref]] = Field(default_factory=dict)

    offer: Ref | None = None
    offer_by: Party = "UNCLEAR"
    acceptance: Ref | None = None
    acceptance_by: Party = "UNCLEAR"

    price_ref: Ref | None = None
    price_amount: int | None = None
    subject_matter_ref: Ref | None = None
    quantity_or_scope_ref: Ref | None = None
    due_date_ref: Ref | None = None
    performance_refs: list[Ref] = Field(default_factory=list)
    # breach_refs: messages where RESPONDENT failed to perform. Do NOT include messages
    # where the respondent accuses the claimant of not paying — those go in
    # scope_dispute_refs (extra-charge demands) or counter_allegation_refs (other accusations).
    breach_refs: list[Ref] = Field(default_factory=list)
    # scope_dispute_refs: messages where a new or extra charge is demanded or disputed
    scope_dispute_refs: list[Ref] = Field(default_factory=list)
    # counter_allegation_refs: messages where the respondent accuses the claimant (non-breach)
    counter_allegation_refs: list[Ref] = Field(default_factory=list)
    # amount_msgs: messages that contain a specific KES amount relevant to the contract
    amount_msgs: list[Ref] = Field(default_factory=list)
    payments: list[PaymentFact] = Field(default_factory=list)

    social_or_domestic_context: bool = False
    capacity_concern_refs: list[Ref] = Field(default_factory=list)
    consent_concern_refs: list[Ref] = Field(default_factory=list)
    illegality_concern_refs: list[Ref] = Field(default_factory=list)
    formality_required: bool = False

    integrity_concerns: list[IntegrityConcern] = Field(default_factory=list)
    element_opinions: dict[str, ElementOpinion] = Field(default_factory=dict)
    narrative: Narrative = Field(default_factory=Narrative)


class VerifyFacts(BaseModel):
    """Partial output from call 2 (verify) — only the decisive fields."""
    model_config = ConfigDict(extra="ignore")

    offer: Ref | None = None
    offer_by: Party = "UNCLEAR"
    acceptance: Ref | None = None
    acceptance_by: Party = "UNCLEAR"
    price_ref: Ref | None = None
    price_amount: int | None = None
    subject_matter_ref: Ref | None = None
    scope_dispute_refs: list[Ref] = Field(default_factory=list)
    payments: list[PaymentFact] = Field(default_factory=list)
