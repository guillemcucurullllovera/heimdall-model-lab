"""Default offline science; original private assertions stay explicitly opt-in."""
import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--private-estate', action='store_true', help='Run original private-data/evidence assertions (requires excluded artifacts)')
    parser.addoption('--network', action='store_true', help='Allow explicitly network-marked provider tests (own credentials required)')


def pytest_collection_modifyitems(config, items):
    records = json.loads((Path(__file__).resolve().parents[1]/'PRIVATE_ESTATE_TESTS.json').read_text())
    private = {r['nodeid']: r['reason'] for r in records['tests']}
    for item in items:
        if item.get_closest_marker('network') and not config.getoption('--network'):
            item.add_marker(pytest.mark.skip(reason='Live provider access is opt-in: --network; not validated by offline CI'))
        if item.nodeid in private and not config.getoption('--private-estate'):
            item.add_marker(pytest.mark.skip(reason=private[item.nodeid]+'; use --private-estate'))
