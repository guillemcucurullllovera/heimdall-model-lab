"""Shared base types for all MODEL-1 contracts.

Every contract in this package is a pydantic v2 model with a stable
`schema_name`/`schema_version` pair so JSON Schema can be exported and
committed (`schemas/*.json`), and so ledger records can identify which
version of a contract they carry (records are frozen once written; a
contract change is a new schema_version, never an in-place edit).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ModelLabBaseModel(BaseModel):
    """Base config shared by every MODEL-1 contract.

    - `extra="forbid"`: a contract is a closed schema; unknown fields are a
      bug (or a schema-version mismatch), never silently accepted.
    - `frozen=True`: contracts are immutable value objects once constructed;
      corrections are new records, never in-place mutation (mirrors the
      ledger's own append-only rule).
    - `use_enum_values=True`: enums serialize as their plain string values,
      which keeps JCS canonicalization straightforward.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
        str_strip_whitespace=False,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _reject_nan_inf_floats(cls, v: Any) -> Any:
        # Contracts route monetary/precision-sensitive values through
        # strings (see canon/jcs.py's documented policy); this guard just
        # catches accidental NaN/Infinity floats early, before they reach
        # canonicalization and produce a confusing NonCanonicalFloatError.
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ValueError(f"NaN/Infinity is not permitted in a contract field: {v!r}")
        return v

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-mode dump suitable for JCS canonicalization / hashing."""
        return self.model_dump(mode="json")
