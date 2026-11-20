"""HTTP-level tests for the address-history forensics endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient


def _z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_history_endpoint_reports_segments_and_unique_owner(
    client: TestClient, clock
) -> None:
    from datetime import timedelta

    lease = client.post(
        "/v1/leases", json={"client_id": "dev-1", "ttl_seconds": 3600}
    ).json()
    issued = datetime.fromisoformat(lease["issued_at"].replace("Z", "+00:00"))
    clock.advance(60)  # let the queried window lie in the past

    response = client.get(
        f"/v1/admin/addresses/{lease['ip']}/history",
        params={
            "start": _z(issued + timedelta(seconds=5)),
            "end": _z(issued + timedelta(seconds=30)),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ip"] == lease["ip"]
    assert body["window_start"] == _z(issued + timedelta(seconds=5))
    assert body["window_end"] == _z(issued + timedelta(seconds=30))
    assert body["unique_owner"] == "dev-1"
    assert len(body["segments"]) == 1
    segment = body["segments"][0]
    assert segment["state"] == "held"
    assert segment["client_id"] == "dev-1"
    assert segment["lease_id"] == lease["lease_id"]
    assert segment["expires_at"] == lease["expires_at"]
    assert "history_started_at" in body and "server_time" in body


def test_history_endpoint_renewal_splits_window(
    client: TestClient, clock
) -> None:
    from datetime import timedelta

    lease = client.post(
        "/v1/leases", json={"client_id": "dev-1", "ttl_seconds": 100}
    ).json()
    issued = datetime.fromisoformat(lease["issued_at"].replace("Z", "+00:00"))
    renewed = client.post(
        f"/v1/leases/{lease['lease_id']}/renew", json={"ttl_seconds": 200}
    ).json()
    renewal_cut = datetime.fromisoformat(
        renewed["expires_at"].replace("Z", "+00:00")
    ) - timedelta(seconds=200)
    window_end = renewal_cut + timedelta(seconds=10)
    clock.advance(60)  # queried window must be in the past

    response = client.get(
        f"/v1/admin/addresses/{lease['ip']}/history",
        params={
            "start": _z(issued + timedelta(milliseconds=500)),
            "end": _z(window_end),
        },
    )
    assert response.status_code == 200
    body = response.json()
    held = [segment for segment in body["segments"] if segment["state"] == "held"]
    assert len(held) == 2
    assert held[0]["expires_at"] == lease["expires_at"]
    assert held[1]["expires_at"] == renewed["expires_at"]
    assert held[0]["to"] == held[1]["from"]
    assert body["unique_owner"] == "dev-1"


def test_history_endpoint_requires_operator_token(tmp_path) -> None:
    from app.config import Settings
    from app.db import Store
    from app.main import create_app

    settings = Settings(
        db_path=str(tmp_path / "leases.db"),
        default_pool_cidr="10.20.0.0/29",
        operator_token="s3cret",
    )
    with TestClient(create_app(settings, store=Store(settings))) as locked:
        url = "/v1/admin/addresses/10.20.0.1/history"
        params = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"}
        assert locked.get(url, params=params).status_code == 401
        ok = locked.get(
            url,
            params=params,
            headers={"X-Operator-Token": "s3cret"},
        )
        assert ok.status_code == 200
        assert ok.json()["segments"][0]["state"] == "unknown"


def test_history_endpoint_rejects_bad_inputs(client: TestClient) -> None:
    url = "/v1/admin/addresses/not-an-ip/history"
    response = client.get(
        url,
        params={"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T01:00:00Z"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_address"

    # start >= end
    response = client.get(
        "/v1/admin/addresses/10.20.0.1/history",
        params={"start": "2026-01-01T01:00:00Z", "end": "2026-01-01T00:00:00Z"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_window"

    # naive (timezone-less) timestamps are refused
    response = client.get(
        "/v1/admin/addresses/10.20.0.1/history",
        params={"start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00"},
    )
    assert response.status_code == 422

    # unparseable timestamps are schema errors with the uniform envelope.
    response = client.get(
        "/v1/admin/addresses/10.20.0.1/history",
        params={"start": "nope", "end": "2026-01-01T01:00:00Z"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_history_endpoint_rejects_future_window(client: TestClient) -> None:
    response = client.get(
        "/v1/admin/addresses/10.20.0.1/history",
        params={
            "start": "2099-01-01T00:00:00Z",
            "end": "2099-01-01T01:00:00Z",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "window_not_past"


def test_history_endpoint_accepts_offset_timezones(client: TestClient) -> None:
    response = client.get(
        "/v1/admin/addresses/10.20.0.1/history",
        params={
            "start": "2026-01-01T02:00:00+02:00",  # == 00:00:00Z
            "end": "2026-01-01T03:00:00+02:00",    # == 01:00:00Z
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["window_start"] == "2026-01-01T00:00:00Z"
    assert body["window_end"] == "2026-01-01T01:00:00Z"
