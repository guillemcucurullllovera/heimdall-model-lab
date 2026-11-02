from __future__ import annotations

import json
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.replay.journal_reader import (
    compute_prefix_hash,
    parse_ndjson_prefix,
    read_prefix_bytes,
)


def test_read_prefix_bytes_returns_exact_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "j.ndjson"
    path.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    assert read_prefix_bytes(path) == b'{"a": 1}\n{"b": 2}\n'


def test_compute_prefix_hash_matches_sha256_bytes() -> None:
    data = b"hello world"
    length, digest = compute_prefix_hash(data)
    assert length == len(data)
    assert digest == sha256_bytes(data)


def test_parse_ndjson_prefix_parses_complete_lines() -> None:
    data = b'{"a": 1}\n{"b": 2}\n'
    records = parse_ndjson_prefix(data)
    assert records == [{"a": 1}, {"b": 2}]


def test_parse_ndjson_prefix_drops_torn_tail_silently() -> None:
    data = b'{"a": 1}\n{"incompl'
    records = parse_ndjson_prefix(data)
    assert records == [{"a": 1}]


def test_parse_ndjson_prefix_skips_blank_lines() -> None:
    data = b'{"a": 1}\n\n{"b": 2}\n'
    records = parse_ndjson_prefix(data)
    assert records == [{"a": 1}, {"b": 2}]


def test_parse_ndjson_prefix_raises_on_non_tail_corruption() -> None:
    data = b"not json\n{\"b\": 2}\n"
    try:
        parse_ndjson_prefix(data)
        raise AssertionError("expected json.JSONDecodeError")
    except json.JSONDecodeError:
        pass
