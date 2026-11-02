from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.data.credentials import (
    MissingCredentialError,
    read_env_credential,
    require_env_credential,
)


def test_env_var_takes_precedence_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "data.env"
    env_file.write_text('FOO_KEY="from-file"\n', encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "from-env")
    assert read_env_credential("FOO_KEY", env_file=env_file) == "from-env"


def test_falls_back_to_file_when_env_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    env_file = tmp_path / "data.env"
    env_file.write_text('FOO_KEY="from-file"\n', encoding="utf-8")
    assert read_env_credential("FOO_KEY", env_file=env_file) == "from-file"


def test_strips_single_and_double_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    env_file = tmp_path / "data.env"
    env_file.write_text("FOO_KEY='single-quoted'\n", encoding="utf-8")
    assert read_env_credential("FOO_KEY", env_file=env_file) == "single-quoted"


def test_ignores_comments_and_blank_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    env_file = tmp_path / "data.env"
    env_file.write_text("# comment\n\nFOO_KEY=bar\n", encoding="utf-8")
    assert read_env_credential("FOO_KEY", env_file=env_file) == "bar"


def test_missing_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    assert read_env_credential("FOO_KEY", env_file=tmp_path / "nonexistent.env") is None


def test_require_raises_missing_credential_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    with pytest.raises(MissingCredentialError):
        require_env_credential("FOO_KEY", env_file=tmp_path / "nonexistent.env")


def test_missing_credential_error_never_contains_a_value() -> None:
    err = MissingCredentialError("SOME_SECRET_VAR")
    assert "SOME_SECRET_VAR" in str(err)
