"""Rich FollowUpRequest model for guided evidence strengthening (v2.0).

Extends the legacy FollowUpRequest in models.py with:
- gap_code: stable catalog key so requests survive rounds
- elements: list of contract elements or s.106B conditions this resolves
- priority: 1 = blocks certificate, 2 = raises confidence, 3 = optional
- kind: UPLOAD | INPUT | CHOICE | CONFIRM
- how: concrete, message-specific instruction
- expected_effect: one-line summary of what resolves

The catalog in followup_catalog.py produces these; the pipeline stores them
under `guided_follow_ups` in the analysis response (separate from the legacy
`result.follow_up_requests` which uses the old schema).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

RequestKind = Literal["UPLOAD", "INPUT", "CHOICE", "CONFIRM"]
UploadType = Literal[
    "CHAT_SCREENSHOT", "MPESA_SMS", "MPESA_STATEMENT",
    "INVOICE_OR_QUOTE", "RECEIPT", "PHOTO", "CONTACT_INFO_SCREENSHOT",
]
InputType = Literal["TEXT", "PHONE", "DATE", "AMOUNT_KES", "MSG_PICK", "YES_NO"]
ReqStatus = Literal["OPEN", "RESOLVED", "STILL_OPEN", "UNAVAILABLE", "SUPERSEDED"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class InputField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    label: str
    type: InputType
    required: bool = False
    options: list[dict[str, Any]] | None = None
    help: str | None = None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FollowUpRequestV2(BaseModel):
    """Rich follow-up request produced by the catalog."""
    model_config = ConfigDict(extra="ignore")

    id: str                           # stable: f"{gap_code}:{target}"
    gap_code: str                     # catalog key e.g. "C_OFFER_MISSING"
    elements: list[str] = Field(default_factory=list)  # ["C"] or ["106B_4a"]
    priority: int = 2                 # 1 blocks certificate, 2 raises confidence, 3 nice-to-have
    blocks_certificate: bool = False
    kind: RequestKind = "UPLOAD"
    title: str = ""                   # ≤ 10 words
    why: str = ""                     # one sentence: gap + why it matters
    how: str = ""                     # concrete instruction with msg_index refs where applicable
    upload_types: list[UploadType] = Field(default_factory=list)
    fields: list[InputField] = Field(default_factory=list)
    example: str | None = None
    related_msg_indices: list[int] = Field(default_factory=list)
    status: ReqStatus = "OPEN"
    expected_effect: str = ""         # e.g. "Can move G from PARTIAL to PRESENT"

    # Legacy bridge: map to old schema fields so existing UI still works
    @property
    def element(self) -> str:
        return self.elements[0] if self.elements else ""

    @property
    def type(self) -> str:  # type: ignore[override]
        """Old schema 'type' field (upload_xxx / provide_xxx / confirm)."""
        if self.kind == "UPLOAD":
            if self.upload_types:
                return f"upload_{self.upload_types[0].lower()}"
            return "upload_chat_screenshot"
        if self.kind == "INPUT":
            if self.fields:
                return self.fields[0].type.lower()
            return "text"
        if self.kind == "CHOICE":
            return "confirm_choice"
        return "confirm"

    @property
    def prompt(self) -> str:
        return self.title

    @property
    def why_it_matters(self) -> str:
        return self.why

    @property
    def blocking(self) -> bool:
        return self.blocks_certificate

    def to_legacy(self) -> dict[str, Any]:
        """Return dict shaped like the old FollowUpRequest for backwards compat."""
        return {
            "id": self.id,
            "element": self.element,
            "type": self.type,
            "prompt": self.title,
            "why_it_matters": self.why,
            "blocking": self.blocks_certificate,
            "target": self.related_msg_indices[0] if self.related_msg_indices else None,
            "options": [f.key for f in self.fields] if self.fields else [],
            "date_role": next(
                (f.type.lower() for f in self.fields if f.type == "DATE"), None
            ),
        }


# ---------------------------------------------------------------------------
# Round diff (returned on supplement responses)
# ---------------------------------------------------------------------------

class ElementChange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    element: str
    from_status: str = Field(alias="from")
    to_status: str = Field(alias="to")
    because: str = ""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class RoundDiff(BaseModel):
    model_config = ConfigDict(extra="ignore")

    round: int
    previous_analysis_id: str | None = None
    confidence: dict[str, float] = Field(default_factory=dict)   # {from, to}
    band: dict[str, str] = Field(default_factory=dict)            # {from, to}
    verdict: dict[str, str] = Field(default_factory=dict)         # {from, to}
    elements_changed: list[dict[str, Any]] = Field(default_factory=list)
    requests_resolved: list[str] = Field(default_factory=list)
    requests_still_open: list[dict[str, Any]] = Field(default_factory=list)
    new_requests: list[str] = Field(default_factory=list)
    cert_eligibility: dict[str, bool] = Field(default_factory=dict)  # {from, to}


# ---------------------------------------------------------------------------
# Statement (typed answer stored as unverified)
# ---------------------------------------------------------------------------

class Statement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field_key: str
    value: str
    round: int = 1
    request_id: str | None = None
    label: str = ""
