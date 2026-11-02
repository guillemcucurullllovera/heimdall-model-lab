"""`model-lab-gate` CLI."""

from __future__ import annotations

from pathlib import Path

import click

from model_lab.gate_v2.v1_freeze import (
    DEFAULT_MANIFEST_PATH,
    verify_v1_freeze,
    write_operator_signing_request,
)


@click.group()
def main() -> None:
    """MODEL-1 Gate v2 CLI."""


@main.command("verify-v1-freeze")
@click.option("--manifest", type=click.Path(exists=True, path_type=Path), default=DEFAULT_MANIFEST_PATH)
@click.option("--write-request", type=click.Path(path_type=Path), default=None, help="write an unsigned operator-signing request to this path")
def verify_v1_freeze_cmd(manifest: Path, write_request: Path | None) -> None:
    result = verify_v1_freeze(manifest)
    click.echo(f"manifest: {result.manifest_path}")
    click.echo(f"files checked: {result.n_files_checked}")
    click.echo(f"violations: {result.n_violations}")
    click.echo(f"all_ok: {result.all_ok}")
    if not result.all_ok:
        for v in result.violations:
            click.echo(f"  VIOLATION {v.path}: manifest={v.manifest_sha256} actual={v.recomputed_sha256} error={v.error}")
    if write_request:
        write_operator_signing_request(result, write_request)
        click.echo(f"wrote unsigned signing request to {write_request}")
    if not result.all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
