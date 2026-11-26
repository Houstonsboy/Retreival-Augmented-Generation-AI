"""Pydantic models for the Section 106B device questionnaire and gate output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProductionMethod(str, Enum):
    SCREENSHOT_ON_DEVICE = "screenshot_on_device"
    FORWARDED_OR_COPIED  = "forwarded_or_copied"
    PRINTED_STATEMENT    = "printed_statement"
    OTHER                = "other"


class DeviceFacts(BaseModel):
    """Litigant-supplied device facts for the Section 106B compliance gate."""

    model_config = ConfigDict(extra="forbid")

    device_make:            str
    device_model:           str
    phone_number:           str | None = Field(default=None, pattern=r"^\+[1-9][0-9]{6,14}$")
    phone_number_last4:     str | None = None   # Legacy certificate records
    production_method:      ProductionMethod
    owner_is_declarant:     bool
    lawful_control:         bool                # s.106B(2)(a)
    regular_use:            bool                # s.106B(2)(a)
    regular_feed:           bool                # s.106B(2)(b)
    operating_properly:     bool                # s.106B(2)(c)
    malfunction_details:    str | None = None   # required when operating_properly is False
    malfunction_affects_record: bool | None = None
    messages_on_this_device: bool               # s.106B(2)(d)
    declarant_name:         str
    declarant_position:     str
    declaration_confirmed:  bool

    @field_validator("device_make", "device_model", "declarant_name", "declarant_position",
                     mode="before")
    @classmethod
    def _strip_nonempty(cls, v):
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("phone_number_last4", mode="before")
    @classmethod
    def _validate_last4(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip()
        if v and (len(v) != 4 or not v.isdigit()):
            raise ValueError("phone_number_last4 must be exactly 4 digits")
        return v or None

    @model_validator(mode="after")
    def _malfunction_requires_details(self) -> "DeviceFacts":
        if not self.operating_properly:
            if not self.malfunction_details or not self.malfunction_details.strip():
                raise ValueError(
                    "malfunction_details is required when operating_properly is False"
                )
            if self.malfunction_affects_record is None:
                raise ValueError(
                    "malfunction_affects_record is required when operating_properly is False"
                )
        return self


class ConditionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code:   str           # "4a", "4b", "2a", "2b", "2c", "2d", "4d", "INTEGRITY"
    title:  str
    met:    bool
    basis:  list[str]     # human-readable reasons
    remedy: str | None    # plain-language fix if not met


class GateResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_type: Literal["CERTIFICATE", "DEFICIENCY_REPORT"]
    outcome:       Literal["certificate_ready", "needs_strengthening", "weak"]
    conditions:    list[ConditionResult]
    unmet:         list[str]    # condition codes not met
    gate_version:  str
