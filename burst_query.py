"""Select saved burst rows by detection strength instead of by time."""

from __future__ import annotations

import pandas as pd


def stations_for_place(instruments: list[str], place: str) -> list[str]:
    """Station names that contain `place`.

    Matching is a case-insensitive substring of the e-Callisto station
    name. "Australia" keeps Australia-ASSA_57 and Australia-ASSA_63.
    Short codes are not mapped to a country: MRO is Metsähovi in Finland,
    so it does not match Australia. An empty place returns every station.
    """
    text = (place or "").strip().lower()
    if not text:
        return list(instruments)
    return [name for name in instruments if text in name.lower()]


def filter_by_place(df: pd.DataFrame, place: str) -> pd.DataFrame:
    """Keep rows whose station name contains `place`."""
    if df.empty:
        return df
    text = (place or "").strip().lower()
    if not text:
        return df
    mask = (
        df["Instrument Location"].astype(str).str.lower().str.contains(text, regex=False)
    )
    return df.loc[mask].copy()


def rank_bursts_by_intensity(
    df: pd.DataFrame, place: str = "", min_stations: int = 1
) -> pd.DataFrame:
    """All matching detections, strongest first.

    Strength is the confidence written into the spectrogram filename
    (the predictor stores probability × 100). It is not a radio flux.
    Time is not the sort key. One row is kept per station and 15-minute
    group: the file with the higher confidence.
    """
    chosen = filter_by_place(df, place)
    if chosen.empty:
        return chosen
    if int(min_stations) > 1 and "TimeGroup" in chosen.columns:
        counts = chosen.groupby("TimeGroup")["Instrument Location"].nunique()
        keep = counts[counts >= int(min_stations)].index
        chosen = chosen[chosen["TimeGroup"].isin(keep)].copy()
        if chosen.empty:
            return chosen
    ordered = chosen.sort_values(
        ["Confidence", "Datetime"], ascending=[False, False], kind="mergesort"
    )
    best = ordered.groupby(
        ["TimeGroup", "Instrument Location"], sort=False
    ).head(1)
    return (
        best.sort_values(
            ["Confidence", "Datetime"], ascending=[False, False], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def select_bursts_by_intensity(
    df: pd.DataFrame,
    place: str = "",
    n: int = 10,
    min_stations: int = 1,
) -> pd.DataFrame:
    """The `n` strongest detections for a place or station name.

    See `rank_bursts_by_intensity` for what "strongest" means.
    """
    ranked = rank_bursts_by_intensity(df, place=place, min_stations=min_stations)
    return ranked.head(max(1, int(n))).reset_index(drop=True)
