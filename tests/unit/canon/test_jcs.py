"""JCS canonicalization tests.

Key-ordering and string-escaping vectors are adapted from the RFC 8785
reference test suite (cyberphone/json-canonicalization); the numeric
sub-vector is intentionally excluded because MODEL-1's canonicalizer
enforces a stricter, documented policy (integral floats only — see
canon/jcs.py module docstring) rather than implementing full ECMAScript
Number::toString.
"""

from __future__ import annotations

import pytest

from model_lab.canon.jcs import (
    NonCanonicalFloatError,
    canonical_sha256,
    canonical_sha256_excluding,
    canonicalize,
)


def test_object_keys_sorted_by_utf16_code_unit() -> None:
    obj = {"b": 1, "a": 2, "c": 3}
    assert canonicalize(obj) == b'{"a":2,"b":1,"c":3}'


def test_nested_objects_and_arrays() -> None:
    obj = {"z": [3, 2, 1], "a": {"y": 1, "x": 2}}
    assert canonicalize(obj) == b'{"a":{"x":2,"y":1},"z":[3,2,1]}'


def test_no_insignificant_whitespace() -> None:
    obj = {"a": 1, "b": [1, 2, 3]}
    out = canonicalize(obj)
    assert b" " not in out
    assert b"\n" not in out


def test_unicode_strings_not_escaped_beyond_mandatory_set() -> None:
    obj = {"currency": "€"}  # euro sign
    out = canonicalize(obj)
    assert "€".encode() in out  # emitted as raw UTF-8, not \u escaped


def test_control_chars_are_escaped() -> None:
    obj = {"s": "a\nb\tc"}
    out = canonicalize(obj).decode("utf-8")
    assert "\\n" in out
    assert "\\t" in out


def test_literals() -> None:
    assert canonicalize(None) == b"null"
    assert canonicalize(True) == b"true"
    assert canonicalize(False) == b"false"


def test_integers_render_plain() -> None:
    assert canonicalize(0) == b"0"
    assert canonicalize(-42) == b"-42"
    assert canonicalize(1234567890123) == b"1234567890123"


def test_integral_float_renders_without_trailing_zero() -> None:
    assert canonicalize(3.0) == b"3"


def test_non_integral_float_raises() -> None:
    with pytest.raises(NonCanonicalFloatError):
        canonicalize(0.1)


def test_nan_and_inf_raise() -> None:
    with pytest.raises(NonCanonicalFloatError):
        canonicalize(float("nan"))
    with pytest.raises(NonCanonicalFloatError):
        canonicalize(float("inf"))


def test_key_order_does_not_change_hash() -> None:
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert canonical_sha256(a) == canonical_sha256(b)


def test_different_values_change_hash() -> None:
    assert canonical_sha256({"x": 1}) != canonical_sha256({"x": 2})


def test_sha256_is_hex64() -> None:
    h = canonical_sha256({"a": 1})
    assert len(h) == 64
    int(h, 16)  # does not raise


def test_canonical_sha256_excluding_self_referential_pattern() -> None:
    # Mirrors gate_v2_spec.json's own convention: a document can embed a
    # hash of itself by excluding the hash field from what gets hashed.
    doc = {"a": 1, "b": 2, "self_hash": "placeholder"}
    h = canonical_sha256_excluding(doc, "self_hash")
    doc["self_hash"] = h
    # Recomputing with the field now populated (still excluded) must be stable.
    assert canonical_sha256_excluding(doc, "self_hash") == h
    # And it must differ from a doc with different content.
    other = {"a": 1, "b": 3, "self_hash": h}
    assert canonical_sha256_excluding(other, "self_hash") != h
