"""Credential loading for MODEL-1's own, isolated provider clients.

Mirrors the estate's own `read_env_credential` pattern (env var first, then a
`NAME=value` line in `~/.config/heimdall/data.env`, values optionally quoted)
but is a fresh, independent implementation — this module imports nothing
from heimdall-v2. Never logs or returns a value in an exception message.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = Path.home() / ".config" / "model-lab" / "data.env"


class MissingCredentialError(RuntimeError):
    def __init__(self, var_name: str) -> None:
        super().__init__(f"credential {var_name!r} not found in environment or {DEFAULT_ENV_FILE}")
        self.var_name = var_name


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def read_env_credential(var_name: str, *, env_file: Path = DEFAULT_ENV_FILE) -> str | None:
    env_value = os.environ.get(var_name)
    if env_value:
        return env_value.strip()
    if not env_file.exists():
        return None
    prefix = f"{var_name}="
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith(prefix):
            return _strip_quotes(line[len(prefix) :])
    return None


def require_env_credential(var_name: str, *, env_file: Path = DEFAULT_ENV_FILE) -> str:
    value = read_env_credential(var_name, env_file=env_file)
    if not value:
        raise MissingCredentialError(var_name)
    return value
