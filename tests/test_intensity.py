"""Rank burst rows by detection confidence and by station name."""

from datetime import datetime

import pandas as pd

from burst_query import (
    rank_bursts_by_intensity,
    select_bursts_by_intensity,
    stations_for_place,
)

INSTRUMENTS = [
    "Australia-ASSA_57",
    "Australia-ASSA_63",
    "MEXART_59",
    "MRO_59",
    "TRIEST_57",
]


def _row(when, station, confidence):
    dt = datetime.fromisoformat(when)
    return {
        "Datetime": dt,
        "Instrument Location": station,
        "Confidence": confidence,
        "Path": f"{confidence:.2f}_{station}.png",
        "TimeGroup": pd.Timestamp(dt).floor("15min"),
    }


def _frame():
    # Times are newest-last on purpose: intensity order must not follow time.
    return pd.DataFrame(
        [
            _row("2026-10-04T08:00:00", "Australia-ASSA_63", 61.0),
            _row("2026-10-03T12:00:00", "Australia-ASSA_57", 88.0),
            _row("2026-10-03T12:00:00", "Australia-ASSA_57", 70.0),
            _row("2026-10-02T18:00:00", "Australia-ASSA_63", 74.0),
            _row("2026-10-04T08:00:00", "MEXART_59", 99.0),
            _row("2026-10-01T00:00:00", "MRO_59", 95.0),
            _row("2026-09-30T00:00:00", "TRIEST_57", 64.0),
        ]
    )


def test_australia_is_the_assa_stations_only():
    found = stations_for_place(INSTRUMENTS, "australia")
    assert found == ["Australia-ASSA_57", "Australia-ASSA_63"]


def test_strongest_australia_ignores_time_and_other_countries():
    ranked = select_bursts_by_intensity(_frame(), place="Australia", n=10)
    assert list(ranked["Instrument Location"]) == [
        "Australia-ASSA_57",
        "Australia-ASSA_63",
        "Australia-ASSA_63",
    ]
    assert list(ranked["Confidence"]) == [88.0, 74.0, 61.0]
    # The weaker file from the same station and quarter-hour is dropped.
    assert 70.0 not in set(ranked["Confidence"])


def test_limit_returns_the_strongest_only():
    ranked = select_bursts_by_intensity(_frame(), place="Australia", n=1)
    assert len(ranked) == 1
    assert ranked.iloc[0]["Confidence"] == 88.0


def test_empty_place_ranks_every_station_by_confidence():
    ranked = rank_bursts_by_intensity(_frame(), place="")
    assert list(ranked["Confidence"]) == [99.0, 95.0, 88.0, 74.0, 64.0, 61.0]


def test_unknown_place_is_empty():
    ranked = select_bursts_by_intensity(_frame(), place="Antarctica", n=5)
    assert ranked.empty


def test_min_stations_keeps_only_groups_seen_at_two_australian_stations():
    extra = pd.DataFrame(
        [_row("2026-10-02T18:00:00", "Australia-ASSA_57", 80.0)]
    )
    frame = pd.concat([_frame(), extra], ignore_index=True)
    ranked = select_bursts_by_intensity(
        frame, place="Australia", n=10, min_stations=2
    )
    assert list(ranked["Confidence"]) == [80.0, 74.0]
    assert set(ranked["Instrument Location"]) == {
        "Australia-ASSA_57",
        "Australia-ASSA_63",
    }
