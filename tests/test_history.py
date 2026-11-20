"""History / address-attribution window tests.

Covers the forensics contract end to end at store level:

* segments split across a renewal and report the agreed ``expires_at`` in
  force at each time;
* natural expiry ends a held segment at the deadline without any sweeper;
* released gaps and re-acquisition are reported as ``unheld``;
* time before the traceability horizon (fresh-DB upgrade instant, or the
  retrofit point of an old-schema database) is ``unknown``;
* ``unique_owner`` is only set when the whole window is one held client;
* history is written in the same transaction as the lease change, so a
  failure leaves nothing behind, and the horizon plus history survive a
  restart.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from app.config import Settings
from app.db import LeaseError, Store, iso


def parse_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def horizon_of(store: Store, ip: str = "10.20.0.1") -> float:
    probe = store.clock()
    return parse_iso(
        store.address_history(ip, probe, probe + 1)["history_started_at"]
    )


def window_before(issued_iso: str, margin: float = 0.5) -> float:
    """A valid window start just before a lease's internal issued instant.

    The fake clock advances one second per call; half a second is inside the
    same tick and above the earlier horizon tick.
    """
    return parse_iso(issued_iso) - margin


def states_of(report: dict) -> list[str]:
    return [segment["state"] for segment in report["segments"]]


# --------------------------------------------------------------------- store


def test_acquire_opens_history_interval(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a")
    clock.advance(5)
    end_at = clock.value
    report = store.address_history(
        lease["ip"], window_before(lease["issued_at"]), end_at
    )
    # The window starts just before the issue, still inside the horizon.
    assert states_of(report) == ["unheld", "held"]
    segment = report["segments"][-1]
    assert segment["state"] == "held"
    assert segment["client_id"] == "client-a"
    assert segment["lease_id"] == lease["lease_id"]
    assert segment["expires_at"] == lease["expires_at"]
    assert segment["from"] == lease["issued_at"]
    assert report["unique_owner"] is None  # the unheld prefix blocks it

    # A window wholly inside the held interval does attribute uniquely.
    inside = store.address_history(
        lease["ip"],
        parse_iso(lease["issued_at"]) + 1,
        parse_iso(lease["issued_at"]) + 10,
    )
    assert inside["unique_owner"] == "client-a"


def test_window_across_renewal_splits_segments(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=100)
    first_expiry = parse_iso(lease["expires_at"])
    clock.advance(40)
    renewed = store.renew(lease["lease_id"], ttl_seconds=200)
    # renew() consumes one clock tick for its effective instant; the second
    # history interval starts at that tick.
    renewal_at = parse_iso(renewed["expires_at"]) - 200
    clock.advance(30)
    end_at = clock.value

    report = store.address_history(
        lease["ip"], window_before(lease["issued_at"]), end_at
    )
    assert states_of(report) == ["unheld", "held", "held"]
    held = [segment for segment in report["segments"] if segment["state"] == "held"]
    assert len(held) == 2
    assert {segment["lease_id"] for segment in held} == {lease["lease_id"]}
    # First segment carries the originally agreed deadline, the second the
    # deadline agreed at renewal time.
    assert parse_iso(held[0]["expires_at"]) == first_expiry
    assert held[1]["expires_at"] == renewed["expires_at"]
    # Half-open adjacency: the renewal instant is where the segments meet.
    assert held[0]["to"] == held[1]["from"] == iso(renewal_at)
    assert parse_iso(held[1]["to"]) == end_at
    assert report["unique_owner"] is None  # unheld prefix blocks attribution

    # A window fully inside the held time (even crossing the renewal cut)
    # attributes the address uniquely to the one client.
    inside = store.address_history(
        lease["ip"],
        parse_iso(lease["issued_at"]) + 5,
        renewal_at + 5,
    )
    assert states_of(inside) == ["held", "held"]
    assert inside["unique_owner"] == "client-a"


def test_natural_expiry_ends_segment_without_sweep(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=60)
    issued = parse_iso(lease["issued_at"])
    deadline = parse_iso(lease["expires_at"])
    clock.advance(120)
    end_at = clock.value
    # No write has happened since expiry, so no sweep could have run; the
    # report must still treat the lease as dead at its agreed deadline.
    report = store.address_history(
        lease["ip"], window_before(lease["issued_at"]), end_at
    )
    states = [
        (segment["state"], parse_iso(segment["to"]))
        for segment in report["segments"]
    ]
    assert states == [
        ("unheld", issued),
        ("held", deadline),
        ("unheld", end_at),
    ]
    assert report["unique_owner"] is None


def test_sweep_marks_expired_history_interval(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=60)
    clock.advance(61)
    store.acquire("client-b")  # any write triggers the sweep
    with store._read() as conn:
        row = conn.execute(
            "SELECT end_reason, ended_at, expires_at FROM lease_history"
        ).fetchone()
    assert row["end_reason"] == "expired"
    assert row["ended_at"] == row["expires_at"]


def test_release_gap_then_reacquire_is_unheld_between(store: Store, clock) -> None:
    first, _ = store.acquire("client-a", ttl_seconds=100)
    ip = first["ip"]
    clock.advance(10)
    released, _ = store.release(first["lease_id"])
    release_time = parse_iso(released["released_at"])
    clock.advance(20)
    second, _ = store.acquire("client-b", ttl_seconds=100)
    assert second["ip"] == ip
    clock.advance(5)
    end_at = clock.value

    report = store.address_history(
        ip, window_before(first["issued_at"]), end_at
    )
    assert states_of(report) == ["unheld", "held", "unheld", "held"]
    held_clients = [
        segment["client_id"]
        for segment in report["segments"]
        if segment["state"] == "held"
    ]
    assert held_clients == ["client-a", "client-b"]
    gap = report["segments"][2]
    assert gap["from"] == iso(release_time)
    assert gap["to"] == second["issued_at"]
    # Two different holders and an unheld gap mean no unique attribution.
    assert report["unique_owner"] is None


def test_same_client_reacquire_keeps_owner_but_not_unique_across_gap(
    store: Store, clock
) -> None:
    first, _ = store.acquire("client-a", ttl_seconds=60)
    ip = first["ip"]
    clock.advance(61)
    # Expiry is automatic at the deadline; the next acquire sweeps and reissues.
    second, reused = store.acquire("client-a", ttl_seconds=60)
    assert reused is False
    assert second["ip"] == ip and second["lease_id"] != first["lease_id"]
    clock.advance(5)
    end_at = clock.value
    report = store.address_history(
        ip, window_before(first["issued_at"]), end_at
    )
    assert report["unique_owner"] is None  # the unheld gap blocks attribution
    assert states_of(report) == [
        "unheld",
        "held",
        "unheld",
        "held",
    ]


def test_unknown_before_horizon_and_held_after(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=100)
    horizon = horizon_of(store, lease["ip"])
    end_at = clock.value
    report = store.address_history(lease["ip"], horizon - 50, end_at)
    assert [segment["state"] for segment in report["segments"]] == [
        "unknown",
        "unheld",
        "held",
    ]
    assert report["segments"][0]["to"] == iso(horizon)
    assert report["segments"][-1]["lease_id"] == lease["lease_id"]
    assert report["unique_owner"] is None


def test_window_fully_before_horizon_is_all_unknown(store: Store) -> None:
    lease, _ = store.acquire("client-a")
    horizon = horizon_of(store, lease["ip"])
    report = store.address_history(lease["ip"], horizon - 100, horizon - 10)
    assert len(report["segments"]) == 1
    assert report["segments"][0]["state"] == "unknown"
    assert report["unique_owner"] is None


def test_never_used_address_is_unheld_after_horizon(store: Store) -> None:
    horizon = horizon_of(store, "10.20.0.7")
    probe = store.clock()
    report = store.address_history("10.20.0.7", horizon + 0.5, probe)
    assert len(report["segments"]) == 1
    assert report["segments"][0]["state"] == "unheld"
    assert report["unique_owner"] is None


def test_unique_owner_only_when_whole_window_held(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=100)
    # A sub-window strictly inside the held interval gets a unique owner.
    issued = parse_iso(lease["issued_at"])
    inside = store.address_history(lease["ip"], issued + 5, issued + 20)
    assert inside["unique_owner"] == "client-a"
    # The same window touching the unheld tail after expiry does not.
    clock.advance(130)
    end_at = clock.value
    outside = store.address_history(lease["ip"], issued + 5, end_at)
    assert outside["unique_owner"] is None


def test_window_validation(store: Store) -> None:
    with pytest.raises(LeaseError) as exc:
        store.address_history("999.1.1.1", 0, 1)
    assert exc.value.code == "invalid_address"

    with pytest.raises(LeaseError) as exc:
        store.address_history("10.20.0.1", 20, 10)
    assert exc.value.code == "invalid_window"

    with pytest.raises(LeaseError) as exc:
        store.address_history("10.20.0.1", 10, 10)
    assert exc.value.code == "invalid_window"


def test_renew_failure_leaves_no_history_behind(store: Store, clock) -> None:
    lease, _ = store.acquire("client-a", ttl_seconds=100)
    store.switch_pool("10.99.0.0/24")  # address now out of pool
    with pytest.raises(LeaseError) as exc:
        store.renew(lease["lease_id"], ttl_seconds=200)
    assert exc.value.code == "address_out_of_pool"
    with store._read() as conn:
        rows = conn.execute(
            "SELECT * FROM lease_history WHERE lease_id = ?", (lease["lease_id"],)
        ).fetchall()
    assert len(rows) == 1  # only the original acquire interval
    assert rows[0]["ended_at"] is None and rows[0]["end_reason"] is None


def test_acquire_failure_rolls_back_history(store: Store) -> None:
    for i in range(6):
        store.acquire(f"client-{i}")
    with pytest.raises(LeaseError) as exc:
        store.acquire("client-overflow")
    assert exc.value.code == "pool_exhausted"
    with store._read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM lease_history").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 6


def test_history_and_horizon_survive_restart(settings: Settings, clock) -> None:
    first = Store(settings, clock=clock)
    first.initialize()
    lease, _ = first.acquire("client-a", ttl_seconds=100)
    clock.advance(10)
    first.renew(lease["lease_id"], ttl_seconds=200)
    horizon_before = horizon_of(first, lease["ip"])

    reopened = Store(settings, clock=clock)
    reopened.initialize()
    horizon_after = horizon_of(reopened, lease["ip"])
    assert horizon_after == horizon_before
    end_at = clock.value
    report = reopened.address_history(
        lease["ip"], window_before(lease["issued_at"]), end_at
    )
    assert states_of(report) == ["unheld", "held", "held"]
    assert report["unique_owner"] is None


# ------------------------------------------------------------- old-database


def _legacy_database(path: str, pool: str, now: float) -> None:
    """Create a database in the pre-history schema, with two leases."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE leases (
                lease_id       TEXT PRIMARY KEY,
                client_id      TEXT NOT NULL,
                ip             TEXT NOT NULL,
                pool_cidr      TEXT NOT NULL,
                issued_at      REAL NOT NULL,
                expires_at     REAL NOT NULL,
                released_at    REAL,
                release_reason TEXT,
                CHECK (released_at IS NULL OR released_at >= issued_at),
                CHECK (expires_at > issued_at),
                CHECK (release_reason IS NULL OR release_reason IN ('released', 'expired'))
            );
            CREATE UNIQUE INDEX ux_leases_active_ip
                ON leases(ip) WHERE released_at IS NULL;
            CREATE UNIQUE INDEX ux_leases_active_client
                ON leases(client_id) WHERE released_at IS NULL;
            CREATE INDEX ix_leases_expires
                ON leases(expires_at) WHERE released_at IS NULL;
            CREATE TABLE pool_changes (
                change_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                previous_cidr  TEXT,
                new_cidr       TEXT NOT NULL,
                changed_at     REAL NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO settings(key, value) VALUES ('current_pool', ?)", (pool,)
        )
        # One lease still valid at the upgrade instant, one already expired.
        conn.execute(
            "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            ("lease-live", "client-live", "10.20.0.1", pool, now - 100, now + 100),
        )
        conn.execute(
            "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            ("lease-dead", "client-dead", "10.20.0.2", pool, now - 300, now - 50),
        )
        conn.commit()
    finally:
        conn.close()


def test_legacy_database_is_retrofitted(tmp_path, clock) -> None:
    db_path = str(tmp_path / "legacy.db")
    upgrade_at = 5_000.0
    _legacy_database(db_path, "10.20.0.0/29", upgrade_at)

    # initialize() consumes one clock tick as the upgrade instant; seed the
    # clock one tick early so the horizon lands exactly on ``upgrade_at``.
    clock.value = upgrade_at - 1
    store = Store(
        Settings(db_path=db_path, default_pool_cidr="10.20.0.0/29"), clock=clock
    )
    store.initialize()

    # The pool is restored rather than reseeded.
    assert store.pool_info()["cidr"] == "10.20.0.0/29"
    probe = clock()
    report = store.address_history(
        "10.20.0.1", upgrade_at - 200, min(upgrade_at + 50, probe)
    )
    assert [segment["state"] for segment in report["segments"]] == [
        "unknown",
        "held",
    ]
    held = report["segments"][1]
    assert held["from"] == iso(upgrade_at)
    assert held["lease_id"] == "lease-live"
    assert held["client_id"] == "client-live"

    # The expired legacy lease is not backfilled: its address reads unheld.
    probe2 = clock()
    dead = store.address_history(
        "10.20.0.2", upgrade_at - 10, min(upgrade_at + 10, probe2)
    )
    assert [segment["state"] for segment in dead["segments"]] == [
        "unknown",
        "unheld",
    ]

    # Restart keeps the same horizon instead of writing a new one.
    again = Store(
        Settings(db_path=db_path, default_pool_cidr="10.20.0.0/29"), clock=clock
    )
    again.initialize()
    assert horizon_of(again, "10.20.0.1") == upgrade_at
