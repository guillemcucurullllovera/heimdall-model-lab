"""RFC 8785 JSON Canonicalization Scheme (JCS) — canonical bytes + hashing.

Used everywhere a hash must be reproducible regardless of key order or
incidental whitespace: ledger record body hashes, gate config hashes
(`gate_config_sha256`), dataset snapshot logical-content hashes, and the
self-referential spec hash pattern used by gate_v2_spec.json
(`spec_sha256_of_this_json_excluding_this_field`).

Scope / known limitation (documented rather than silently wrong): full
RFC 8785 requires ECMAScript Number::toString formatting for JSON numbers,
which differs from Python's `repr(float)` in edge cases (very large/small
magnitudes, some fractional values). MODEL-1's own contract, per
MODEL1_foundation_contracts.md, is to represent monetary/precision-sensitive
values as *strings*, never as JSON floats, specifically to avoid this class
of ambiguity. This module therefore implements the parts of JCS that matter
for our own payloads exactly (recursive key sorting by UTF-16 code unit,
UTF-8 encoding, no insignificant whitespace, `/` not escaped, canonical
string escaping) and applies the ECMAScript-compatible formatter for the
JSON number types we do allow (plain integers). A payload containing a
non-integer JSON float raises `NonCanonicalFloatError` rather than silently
producing a value that would not match another RFC-8785-conformant
implementation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


class NonCanonicalFloatError(ValueError):
    """Raised when canonicalizing a JSON float that RFC 8785 number
    formatting cannot be guaranteed to round-trip identically to other
    conformant implementations. Callers should represent such values as
    strings instead (this is MODEL-1's own contract for monetary/
    precision-sensitive fields)."""


def _canon_string(s: str) -> str:
    # json.dumps with ensure_ascii=False already produces RFC 8785-compatible
    # string escaping for the mandatory-escape set (", \, and control chars
    # < 0x20); it does not escape '/' or non-ASCII, which JCS also requires.
    import json as _json

    return _json.dumps(s, ensure_ascii=False)


def _canon_number(n: int | float) -> str:
    if isinstance(n, bool):  # bool is a subclass of int; never a JSON number here
        raise TypeError("bool is not a JCS number")
    if isinstance(n, int):
        return str(n)
    # float
    if math.isnan(n) or math.isinf(n):
        raise NonCanonicalFloatError(f"NaN/Infinity is not valid JSON: {n!r}")
    if n == int(n) and abs(n) < 1e15:
        # Integral-valued float: ECMAScript Number::toString renders these
        # without a trailing '.0' (e.g. 3.0 -> "3"), matching JCS.
        return str(int(n))
    raise NonCanonicalFloatError(
        f"Non-integral float {n!r} cannot be canonicalized without a full "
        "ECMAScript Number::toString implementation. Represent this value "
        "as a string in the payload instead (MODEL-1's own money/precision "
        "convention)."
    )


def _canon_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _canon_number(v)
    if isinstance(v, str):
        return _canon_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canon_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return _canon_object(v)
    raise TypeError(f"Value of type {type(v)!r} is not JSON-serializable for JCS canonicalization")


def _canon_object(obj: dict[str, Any]) -> str:
    # RFC 8785 §3.2.3: sort object members by UTF-16 code unit of the key.
    # Python str comparison is by Unicode code point, which matches UTF-16
    # code-unit order for the BMP; for astral characters JCS specifies
    # surrogate-pair code-unit order, which we approximate via a UTF-16BE
    # encode-and-compare (exact for all inputs, including astral chars).
    def _utf16_key(k: str) -> bytes:
        return k.encode("utf-16-be", errors="surrogatepass")

    items = sorted(obj.items(), key=lambda kv: _utf16_key(kv[0]))
    parts = [f"{_canon_string(k)}:{_canon_value(v)}" for k, v in items]
    return "{" + ",".join(parts) + "}"


def canonicalize(obj: Any) -> bytes:
    """Return the RFC 8785 canonical JSON bytes (UTF-8) for `obj`.

    `obj` must already be a plain JSON-compatible structure (dict/list/str/
    int/float/bool/None) — pydantic models should be dumped via
    `model.model_dump(mode="json")` first.
    """
    return _canon_value(obj).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    """SHA-256 hex digest of the RFC 8785 canonical JSON bytes of `obj`."""
    return hashlib.sha256(canonicalize(obj)).hexdigest()


def canonical_sha256_excluding(obj: dict[str, Any], excluded_field: str) -> str:
    """Self-referential hash pattern: hash the canonical JSON of `obj` with
    `excluded_field` removed. Mirrors gate_v2_spec.json's own
    `spec_sha256_of_this_json_excluding_this_field` convention, so a document
    can embed a hash of itself.
    """
    trimmed = {k: v for k, v in obj.items() if k != excluded_field}
    return canonical_sha256(trimmed)


def sha256_bytes(data: bytes) -> str:
    """Plain SHA-256 hex digest of raw bytes (for content-addressed raw
    payload storage — no canonicalization, the bytes are hashed as-is)."""
    return hashlib.sha256(data).hexdigest()
