from __future__ import annotations

from datetime import date

import httpx
import pytest

from model_lab.data.providers import massive as massive_mod
from model_lab.data.providers.massive import MassiveClient, MassiveEntitlementError


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(massive_mod, "require_env_credential", lambda name: "fake-key")


def test_fetch_daily_bars_parses_epoch_ms_and_stringifies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "OK", "results": [{"t": 1704171600000, "o": 470.1, "h": 471, "l": 469, "c": 470.5, "v": 100}]},
        )

    client = MassiveClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=massive_mod.BASE_URL)
    result = client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result.bars) == 1
    assert result.bars[0].o == "470.1"


def test_not_authorized_status_raises_entitlement_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "NOT_AUTHORIZED", "results": []})

    client = MassiveClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=massive_mod.BASE_URL)
    with pytest.raises(MassiveEntitlementError):
        client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))


def test_403_with_plan_message_classified_as_entitlement_not_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "upgrade your plan to access this endpoint"})

    client = MassiveClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=massive_mod.BASE_URL)
    with pytest.raises(MassiveEntitlementError):
        client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
