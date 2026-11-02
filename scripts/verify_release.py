"""Verify implementation coverage and publication boundaries, without network access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    coverage = json.loads((ROOT/'SOURCE_COVERAGE.json').read_text())
    retained = {r['destination_path']: r for r in coverage['retained']}
    for name, record in retained.items():
        path = ROOT/name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['destination_sha256'], name
    for record in coverage['source_inventory']:
        if record['path'].startswith('src/'):
            assert record['destination'] in retained, record
        if record['disposition'] == 'EXCLUDED':
            assert record['reason'], record
    modules = list((ROOT/'src/model_lab').rglob('*.py'))
    assert len(modules) >= 144
    forbidden = ('/home/', '/root/', '/srv/', '/opt/', 'BEGIN OPENSSH PRIVATE KEY',
                 'BEGIN RSA PRIVATE KEY')
    for base in ('src', 'tests', 'scripts', 'schemas'):
        for path in (ROOT/base).rglob('*'):
            if not path.is_file() or path.suffix not in ('.py', '.json') or path == Path(__file__).resolve():
                continue
            text = path.read_text()
            assert not any(needle in text for needle in forbidden), str(path.relative_to(ROOT))
    print(f'PASS: {len(modules)} source modules, {len(retained)} retained source files; all branch implementation paths accounted for')


if __name__ == '__main__':
    main()
