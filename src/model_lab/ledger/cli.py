"""`model-lab-ledger` CLI: verify a ledger file, repair a torn tail, and
show the current head hash."""

from __future__ import annotations

from pathlib import Path

import click

from model_lab.ledger.chain import chain_hash
from model_lab.ledger.storage import LedgerFile, repair_torn_tail
from model_lab.ledger.verifier import verify_chain


@click.group()
def main() -> None:
    """MODEL-1 evidence ledger CLI."""


@main.command()
@click.argument("ledger_path", type=click.Path(exists=True, path_type=Path))
def verify(ledger_path: Path) -> None:
    """Verify a ledger file's hash chain end-to-end."""
    records = LedgerFile(ledger_path).read_all()
    result = verify_chain(records)
    click.echo(f"records: {result.n_records}")
    click.echo(f"ok: {result.ok}")
    if not result.ok:
        for v in result.violations:
            click.echo(f"  VIOLATION seq={v.seq} kind={v.kind}: {v.detail}")
        click.echo(f"first_unverifiable_seq: {result.first_unverifiable_seq}")
        if result.model_versions_to_quarantine:
            click.echo(f"model_versions_to_quarantine: {result.model_versions_to_quarantine}")
        raise SystemExit(1)


@main.command()
@click.argument("ledger_path", type=click.Path(exists=True, path_type=Path))
def head(ledger_path: Path) -> None:
    """Print the current head hash (chain_hash of the last record)."""
    records = LedgerFile(ledger_path).read_all()
    if not records:
        click.echo("empty ledger; no head")
        return
    click.echo(chain_hash(records[-1]))


@main.command("repair-torn-tail")
@click.argument("ledger_path", type=click.Path(exists=True, path_type=Path))
def repair_torn_tail_cmd(ledger_path: Path) -> None:
    """Truncate a torn (incomplete) last line, if present."""
    repair = repair_torn_tail(ledger_path)
    if repair is None:
        click.echo("no torn tail found")
    else:
        click.echo(f"repaired: dropped {len(repair.dropped_bytes)} bytes at line index {repair.dropped_at_line_index}")


if __name__ == "__main__":
    main()
