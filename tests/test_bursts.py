"""Filename parsing and the latest-tab listing rule."""

from datetime import datetime

from bursts import (
    LATEST_MIN_STATIONS,
    confidence_percent,
    parse_burst_path,
    rows_for_day,
)
from instruments import INSTRUMENT_LIST

# 1x1 PNG. The website only needs a file the parser accepts.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8cfff1f0005fe02fe0d1322410000000049454e44ae426082"
)


def test_percent_prefix_stays_percent():
    assert confidence_percent(86.5) == 86.5


def test_fraction_prefix_becomes_percent():
    assert confidence_percent(0.865) == 86.5


def test_forty_percent_fails_a_fifty_percent_cut():
    # Comparing the raw prefix 40 to a 0.5 slider would keep this file.
    assert confidence_percent(40.0) < 50


def test_parse_station_with_hyphen_and_underscore():
    row = parse_burst_path(
        "/data/2026/10/03/SRI-Lanka_59/86.50_SRI-Lanka_59_03-10-2026_17-45-00.png"
    )
    assert row is not None
    assert row["Instrument Location"] == "SRI-Lanka_59"
    assert row["Confidence"] == 86.5
    assert row["Datetime"] == datetime(2026, 10, 3, 17, 45, 0)


def test_one_station_day_is_returned(tmp_path):
    day = tmp_path / "2026" / "10" / "03" / "SRI-Lanka_59"
    day.mkdir(parents=True)
    (day / "86.50_SRI-Lanka_59_03-10-2026_17-45-00.png").write_bytes(_PNG)
    low = tmp_path / "2026" / "10" / "03" / "TRIEST_57"
    low.mkdir()
    (low / "40.00_TRIEST_57_03-10-2026_17-45-00.png").write_bytes(_PNG)

    rows = rows_for_day(str(tmp_path), datetime(2026, 10, 3), 0.5, INSTRUMENT_LIST)
    assert [r["Instrument Location"] for r in rows] == ["SRI-Lanka_59"]


def test_latest_tab_lists_a_single_station():
    assert LATEST_MIN_STATIONS == 1
