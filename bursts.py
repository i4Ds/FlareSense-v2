"""Read burst PNG filenames. No Gradio or torch imports."""

import glob
import os
from datetime import datetime
from typing import Dict, List, Optional

# Latest tab on the public site. One station is enough to list a flare:
# requiring three hid the October 2026 detections, which were almost all
# single-station.
LATEST_DAYS = 3
LATEST_MIN_PROBA = 0.5
LATEST_MIN_STATIONS = 1


def confidence_percent(raw: float) -> float:
    """Return 0–100.

    The predictor saves `probability * 100` in the filename (`86.50_...`).
    A value of 1 or less is treated as a fraction (`0.86` → 86).
    """
    if raw > 1:
        return raw
    return raw * 100.0


def parse_burst_path(path: str) -> Optional[Dict]:
    """Parse one PNG path. Return None when the name does not match."""
    base = os.path.basename(path)
    parts = base.split("_")
    try:
        raw = float(parts[0])
        antenna = "_".join(parts[1:-2])
        dt_str = parts[-2] + " " + parts[-1].replace(".png", "")
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")
    except (ValueError, IndexError):
        return None
    if not antenna:
        return None
    return {
        "Datetime": dt,
        "Instrument Location": antenna,
        "Confidence": confidence_percent(raw),
        "Path": path,
    }


def rows_for_day(
    base_path: str,
    date_obj: datetime,
    min_proba: float,
    instruments: List[str],
) -> List[Dict]:
    """PNG rows for one UTC day at or above `min_proba` (0–1)."""
    year = str(date_obj.year)
    month = f"{date_obj.month:02d}"
    day = f"{date_obj.day:02d}"
    search_path = os.path.join(base_path, year, month, day, "*", "*")
    threshold = min_proba * 100.0
    rows = []
    for path in glob.glob(search_path):
        row = parse_burst_path(path)
        if row is None:
            continue
        if row["Confidence"] < threshold:
            continue
        if row["Instrument Location"] not in instruments:
            continue
        rows.append(row)
    return rows
