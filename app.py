import os
import glob
import zipfile
import tempfile
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# -----------------------------
# CONFIG
# -----------------------------
# Symbolic base path for data storage
BASE_PATH = os.path.join("/srv/flaresense-data/burst_live_images")
# Hidden path for UI links
# Basically, a mounted version without the full filesystem path
TIMEGROUP_MINUTES = 15
DEFAULT_MIN_PROBA = 0.5
DEFAULT_MIN_STATIONS = 3

# Your existing list of valid instruments
from pred_live import INSTRUMENT_LIST


# -----------------------------
# UTILITIES
# -----------------------------
def floor_to_group(dt: pd.Timestamp, minutes: int = TIMEGROUP_MINUTES) -> pd.Timestamp:
    return dt.floor(f"{minutes}min")


def today_date() -> datetime:
    return datetime.utcnow()


def ymd_from_date(d: datetime) -> Tuple[str, str, str]:
    return str(d.year), f"{d.month:02d}", f"{d.day:02d}"


def parse_date_text(s: str) -> Optional[datetime]:
    """
    Robust-ish date parser for common formats, including '9..1.2025', '9/1/2025', '2025-01-09',
    'today', 'yesterday', etc. Returns a datetime at 00:00 UTC.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip().lower()

    if s in {"today", "now"}:
        d = today_date()
        return datetime(d.year, d.month, d.day)
    if s in {"yesterday"}:
        d = today_date() - timedelta(days=1)
        return datetime(d.year, d.month, d.day)

    # Pull out number chunks to handle weird separators like '9..1.2025'
    nums = [n for n in re_split_non_digits(s)]
    if len(nums) == 3:
        # Try D-M-Y order (common in EU)
        try:
            d, m, y = int(nums[0]), int(nums[1]), int(nums[2])
            if y < 100:  # support 2-digit years as 20xx
                y += 2000
            return datetime(y, m, d)
        except ValueError:
            pass
        # Try Y-M-D order
        try:
            y, m, d = int(nums[0]), int(nums[1]), int(nums[2])
            if y < 100:
                y += 2000
            return datetime(y, m, d)
        except ValueError:
            pass

    # Last attempt: let pandas try (handles many formats)
    try:
        ts = pd.to_datetime(s, utc=False)  # naive local-like; we treat as UTC day
        return datetime(ts.year, ts.month, ts.day)
    except Exception:
        return None


def re_split_non_digits(s: str) -> List[str]:
    import re

    return [x for x in re.split(r"\D+", s) if x]


# -----------------------------
# DATA LOADING & TRANSFORMATIONS
# -----------------------------
def load_image_paths_for_date(date_obj: datetime, min_proba: float) -> pd.DataFrame:
    """
    Scan directory for one day. Returns DataFrame:
    ['Datetime','Instrument Location','Confidence','Path','TimeGroup'].
    """
    year, month, day = ymd_from_date(date_obj)
    search_path = os.path.join(BASE_PATH, year, month, day, "*", "*")

    table_data = []
    for f in glob.glob(search_path):
        try:
            base = os.path.basename(f)
            parts = base.split("_")
            proba = float(parts[0])
            if proba < min_proba:
                continue

            antenna = "_".join(parts[1:-2])
            dt_str = parts[-2] + " " + parts[-1].replace(".png", "")
            dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")

            # filter invalid instruments early
            if antenna not in INSTRUMENT_LIST:
                continue

            table_data.append([dt, antenna, proba, f])
        except (ValueError, IndexError):
            # malformed filenames are ignored
            continue

    if not table_data:
        return pd.DataFrame(
            columns=[
                "Datetime",
                "Instrument Location",
                "Confidence",
                "Path",
                "TimeGroup",
            ]
        )

    df = (
        pd.DataFrame(
            table_data,
            columns=["Datetime", "Instrument Location", "Confidence", "Path"],
        )
        .sort_values(by="Datetime", ascending=True)
        .reset_index(drop=True)
    )
    df["TimeGroup"] = pd.to_datetime(df["Datetime"]).dt.floor(f"{TIMEGROUP_MINUTES}min")
    return df


def filter_by_min_stations(df: pd.DataFrame, min_stations: int) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby("TimeGroup")["Instrument Location"].nunique()
    valid_groups = counts[counts >= min_stations].index
    return df[df["TimeGroup"].isin(valid_groups)].copy()


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize groups for UI selection. Columns:
    ['GroupStartUTC','Stations','Detections','MaxConfidence']
    """
    if df.empty:
        return pd.DataFrame(
            columns=["GroupStartUTC", "Stations", "Detections", "MaxConfidence"]
        )

    g = df.groupby("TimeGroup").agg(
        Stations=("Instrument Location", "nunique"),
        Detections=("Path", "count"),
        MaxConfidence=("Confidence", "max"),
    )
    g = g.reset_index().rename(columns={"TimeGroup": "GroupStartUTC"})
    g = g.sort_values("GroupStartUTC", ascending=False).reset_index(drop=True)
    return g


def images_for_group(df: pd.DataFrame, group_ts: pd.Timestamp) -> List[Tuple[str, str]]:
    """
    Return a list of (path, caption) for a given time group, sorted by confidence desc.
    """
    if df.empty:
        return []
    group_df = (
        df[df["TimeGroup"] == group_ts]
        .sort_values(by="Confidence", ascending=False)
        .copy()
    )
    out = []
    for _, row in group_df.iterrows():
        cap = f"{row['Instrument Location']} – {row['Datetime'].strftime('%Y-%m-%d %H:%M:%S')} – {row['Confidence']:.1f}%"
        out.append((row["Path"], cap))
    return out


def concatenate_days(days_back: int, min_proba: float) -> pd.DataFrame:
    """
    Load last N days and concat.
    """
    dfs = []
    for i in range(days_back):
        d = today_date() - timedelta(days=i)
        df = load_image_paths_for_date(datetime(d.year, d.month, d.day), min_proba)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(
            columns=[
                "Datetime",
                "Instrument Location",
                "Confidence",
                "Path",
                "TimeGroup",
            ]
        )
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# EXPORTS
# -----------------------------
def create_zip_export_for_date(
    date_obj: datetime, min_proba: float, min_stations: int
) -> Optional[str]:
    table_data = load_image_paths_for_date(date_obj, min_proba)
    if table_data.empty:
        return None

    table_data = filter_by_min_stations(table_data, min_stations)
    if table_data.empty:
        return None

    tmp_dir = tempfile.mkdtemp()
    y, m, d = ymd_from_date(date_obj)
    zip_path = os.path.join(tmp_dir, f"FlareSense_Export_{y}_{m}_{d}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # CSV (remove Path column for privacy/security)
        csv_data = table_data.drop(columns=["TimeGroup", "Path"]).round(2)
        csv_path = os.path.join(tmp_dir, f"bursts_{y}_{m}_{d}.csv")
        csv_data.to_csv(csv_path, index=False)
        zipf.write(csv_path, os.path.basename(csv_path))

        # Images
        for _, row in table_data.iterrows():
            img_path = row["Path"]
            if os.path.exists(img_path):
                burst_time = row["Datetime"].strftime("%H-%M-%S")
                burst_date = row["Datetime"].strftime("%Y-%m-%d")
                confidence = row["Confidence"]
                instrument = row["Instrument Location"]
                filename = (
                    f"{burst_date}_{burst_time}_{instrument}_{confidence:.1f}pct.png"
                )
                zipf.write(img_path, f"images/{filename}")

    return zip_path


# -----------------------------
# PLOTS (Plotly-only: no PNG conversion)
# -----------------------------
def plot_daily_by_station(days: int = 30, min_proba: float = DEFAULT_MIN_PROBA):
    df_all = concatenate_days(days, min_proba)
    if df_all.empty:
        return go.Figure()

    df_all["Day"] = pd.to_datetime(df_all["Datetime"]).dt.floor("D")
    daily_by_station = (
        df_all.groupby(["Day", "Instrument Location"])
        .size()
        .reset_index(name="Count")
        .sort_values(by=["Day", "Count"], ascending=[True, False])
    )

    daily_counts = df_all.groupby("Day").size().reset_index(name="Total Count")
    daily_counts["MA_7"] = daily_counts["Total Count"].rolling(window=7).mean()

    fig = px.bar(
        daily_by_station,
        x="Day",
        y="Count",
        color="Instrument Location",
        barmode="stack",
        title=f"Number of Bursts per Day by Station (Last {days} Days)",
        labels={"Day": "Date", "Count": "Bursts"},
    )
    fig.add_trace(
        go.Scatter(
            x=daily_counts["Day"],
            y=daily_counts["MA_7"],
            mode="lines",
            name="7-Day MA",
        )
    )
    fig.update_layout(
        hovermode="closest",
        xaxis_title="Date",
        yaxis_title="Bursts Detected",
        font=dict(size=14),
    )
    return fig


def plot_all_data_with_ma():
    """
    Plot last 60 days of data with 7-day moving average.
    Fast and simple without caching.
    """
    days_to_show = 60
    today = today_date()

    # Fast filesystem walk for last 60 days only
    daily_data = []

    for i in range(days_to_show):
        current_date = today - timedelta(days=i)
        year, month, day = ymd_from_date(current_date)
        day_path = os.path.join(BASE_PATH, year, month, day)

        if not os.path.exists(day_path):
            continue

        # Count files per station for this day
        for station_dir in os.listdir(day_path):
            station_path = os.path.join(day_path, station_dir)
            if not os.path.isdir(station_path) or station_dir not in INSTRUMENT_LIST:
                continue

            # Count PNG files in this station directory
            try:
                png_count = len(
                    [f for f in os.listdir(station_path) if f.endswith(".png")]
                )
                if png_count > 0:
                    daily_data.append(
                        {
                            "Day": current_date,
                            "Instrument Location": station_dir,
                            "Count": png_count,
                        }
                    )
            except (OSError, PermissionError):
                continue

    if not daily_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the last 60 days",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Convert to DataFrame and process
    df = pd.DataFrame(daily_data)
    df["Day"] = pd.to_datetime(df["Day"])

    # Calculate daily totals and 7-day MA
    daily_counts = df.groupby("Day")["Count"].sum().reset_index()
    daily_counts.columns = ["Day", "Total Count"]
    daily_counts["MA_7"] = (
        daily_counts["Total Count"].rolling(window=7, center=False).mean()
    )

    # Create stacked bar chart
    fig = px.bar(
        df,
        x="Day",
        y="Count",
        color="Instrument Location",
        barmode="stack",
        title="Solar Radio Bursts (Last 60 Days) with 7-Day Moving Average",
        labels={"Day": "Date", "Count": "Bursts"},
    )

    # Add 7-day moving average line (hidden from legend)
    fig.add_trace(
        go.Scatter(
            x=daily_counts["Day"],
            y=daily_counts["MA_7"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(color="red", width=3),
            yaxis="y2",
            showlegend=False,
        )
    )

    # Update layout for dual y-axis and move legend to right
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Daily Burst Detections",
        yaxis2=dict(title="", overlaying="y", side="right", showgrid=False),
        font=dict(size=12),
        height=600,
        width=1600,
        showlegend=True,
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position to the right of the plot
        ),
        margin=dict(r=500),  # Add right margin for legend
    )

    # Configure for better interactivity
    fig.update_layout(
        modebar_add=["select2d", "lasso2d"], modebar_remove=["pan2d", "autoScale2d"]
    )

    return fig


def generate_burst_statistics():
    """
    Generate statistics for the last 60 days.
    """
    days_to_show = 60
    today = today_date()

    # Fast count for last 60 days
    total_bursts = 0
    instrument_counts = {}
    days_with_data = set()

    for i in range(days_to_show):
        current_date = today - timedelta(days=i)
        year, month, day = ymd_from_date(current_date)
        day_path = os.path.join(BASE_PATH, year, month, day)

        if not os.path.exists(day_path):
            continue

        day_has_data = False
        for station_dir in os.listdir(day_path):
            station_path = os.path.join(day_path, station_dir)
            if not os.path.isdir(station_path) or station_dir not in INSTRUMENT_LIST:
                continue

            try:
                png_count = len(
                    [f for f in os.listdir(station_path) if f.endswith(".png")]
                )
                if png_count > 0:
                    total_bursts += png_count
                    instrument_counts[station_dir] = (
                        instrument_counts.get(station_dir, 0) + png_count
                    )
                    day_has_data = True
            except (OSError, PermissionError):
                continue

        if day_has_data:
            days_with_data.add(current_date.date())

    if total_bursts == 0:
        return "<p>No data available for statistics.</p>"

    # Calculate statistics
    total_days_with_data = len(days_with_data)
    avg_bursts_per_day = (
        total_bursts / total_days_with_data if total_days_with_data > 0 else 0
    )

    # Recent activity (last 7 days)
    recent_cutoff = today - timedelta(days=7)
    recent_days = [d for d in days_with_data if d >= recent_cutoff.date()]
    recent_avg = (len(recent_days) * avg_bursts_per_day) / 7 if recent_days else 0

    # Sort instruments by count
    sorted_instruments = sorted(
        instrument_counts.items(), key=lambda x: x[1], reverse=True
    )

    html = f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px;'>
        <h3 style='color: #333; margin-top: 0;'>📊 Burst Detection Statistics (Last 60 Days)</h3>
        
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;'>
            <div style='background: white; padding: 15px; border-radius: 5px; text-align: center;'>
                <h4 style='margin: 0; color: #007bff;'>{total_bursts:,}</h4>
                <p style='margin: 5px 0; color: #666;'>Total Burst Images</p>
            </div>
            <div style='background: white; padding: 15px; border-radius: 5px; text-align: center;'>
                <h4 style='margin: 0; color: #28a745;'>{avg_bursts_per_day:.1f}</h4>
                <p style='margin: 5px 0; color: #666;'>Avg Images/Day</p>
            </div>
            <div style='background: white; padding: 15px; border-radius: 5px; text-align: center;'>
                <h4 style='margin: 0; color: #ffc107;'>{total_days_with_data}</h4>
                <p style='margin: 5px 0; color: #666;'>Days with Data</p>
            </div>
            <div style='background: white; padding: 15px; border-radius: 5px; text-align: center;'>
                <h4 style='margin: 0; color: #dc3545;'>{len(instrument_counts)}</h4>
                <p style='margin: 5px 0; color: #666;'>Active Instruments</p>
            </div>
        </div>
        
        <h4 style='color: #333; margin-bottom: 10px;'>🏢 Detection Images by Instrument:</h4>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;'>
    """

    for instrument, count in sorted_instruments[:15]:  # Top 15 instruments
        percentage = (count / total_bursts) * 100
        html += f"""
            <div style='background: white; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between;'>
                <span style='color: #333;'>{instrument}</span>
                <span style='color: #666; font-weight: bold;'>{count:,} ({percentage:.1f}%)</span>
            </div>
        """

    html += """
        </div>
    </div>
    """

    return html


# -----------------------------
# UI CALLBACKS (pure functions)
# -----------------------------
def _browser_day_core(
    date_text: str,
    days_back_slider: int,
    min_proba: float,
    min_stations: int,
    prefer_slider_date: bool,
):
    """
    Core browser function: determine date from either text or slider (prefer slider if asked),
    load day df, derive group summary, return UI updates.
    """
    # Determine date
    target_date = None
    if prefer_slider_date:
        d = today_date() - timedelta(days=days_back_slider)
        target_date = datetime(d.year, d.month, d.day)
    else:
        parsed = parse_date_text(date_text)
        if parsed is not None:
            target_date = parsed
        else:
            # fallback to slider if text invalid
            d = today_date() - timedelta(days=days_back_slider)
            target_date = datetime(d.year, d.month, d.day)

    # Load and filter
    df = load_image_paths_for_date(target_date, min_proba)
    df = filter_by_min_stations(df, min_stations)

    # Prepare outputs
    if df.empty:
        summary_html = (
            f"<div style='background:#fff3cd;padding:10px;border-radius:6px;'>"
            f"No bursts found for <b>{target_date.strftime('%Y-%m-%d')}</b> "
            f"(min_conf ≥ {min_proba:.1f}, min_stations ≥ {min_stations}).</div>"
        )
        groups_df = pd.DataFrame(
            columns=["GroupStartUTC", "Stations", "Detections", "MaxConfidence"]
        )
        group_choices = []
        default_choice = None
        gallery_items = []
    else:
        groups_df = summarize_groups(df)
        total_dets = len(df)
        uniq_stations = df["Instrument Location"].nunique()
        time_groups = groups_df.shape[0]
        summary_html = (
            f"<div style='background:#d4edda;padding:10px;border-radius:6px;'>"
            f"<b>{target_date.strftime('%Y-%m-%d')}</b>: {total_dets} detections · "
            f"{uniq_stations} station(s) · {time_groups} group(s)</div>"
        )

        # choices for group selector
        def label_row(r):
            t = pd.to_datetime(r["GroupStartUTC"])
            return f"{t.strftime('%Y-%m-%d %H:%M')}  |  {int(r['Stations'])} stations, {int(r['Detections'])} detections, max {r['MaxConfidence']:.1f}%"

        group_choices = [
            (label_row(row), pd.to_datetime(row["GroupStartUTC"]).isoformat())
            for _, row in groups_df.iterrows()
        ]
        default_choice = group_choices[0][1] if group_choices else None
        # initial gallery
        if default_choice:
            gallery_items = images_for_group(df, pd.to_datetime(default_choice))
        else:
            gallery_items = []

    # Display date as canonical YYYY-MM-DD in the textbox
    canonical_text = target_date.strftime("%Y-%m-%d")
    return (
        canonical_text,
        summary_html,
        groups_df,
        gr.update(
            choices=[c[0] for c in group_choices],
            value=(group_choices[0][0] if group_choices else None),
        ),
        # we keep a parallel hidden map value->iso for reliable selection
        [c[1] for c in group_choices],  # hidden list of ISO values in the same order
        gallery_items,
    )


def browser_day_from_text(
    date_text: str, days_back_slider: int, min_proba: float, min_stations: int
):
    return _browser_day_core(
        date_text, days_back_slider, min_proba, min_stations, prefer_slider_date=False
    )


def browser_day_from_slider(
    days_back_slider: int, date_text: str, min_proba: float, min_stations: int
):
    return _browser_day_core(
        date_text, days_back_slider, min_proba, min_stations, prefer_slider_date=True
    )


def update_gallery_for_selected_group(
    selected_label: str,
    hidden_iso_list: List[str],
    current_date_text: str,
    min_proba: float,
    min_stations: int,
):
    """
    selected_label is the label shown in the dropdown. We map it to the same index in hidden_iso_list
    to get the ISO timestamp. Then we reload the day's df and extract images.
    """
    if not selected_label or not hidden_iso_list:
        return []

    # Map label -> index using order; Gradio returns the label string
    # The label list is identical in order to hidden_iso_list
    # We find index by extracting HH:MM from label and matching iso list; safer approach:
    # We'll compute index by direct position lookup from the dropdown choices, which we don't have here.
    # Easier: pass the index as part of the label "… [#i]"? Instead, do a fuzzy parse of timestamp at start.
    # Labels start with "YYYY-MM-DD HH:MM  | ..."
    try:
        ts_str = selected_label.split("|")[0].strip()  # "YYYY-MM-DD HH:MM"
        # add ":00" seconds
        ts_str = ts_str + ":00"
        group_ts = pd.to_datetime(ts_str)
    except Exception:
        return []

    # Load the date dataframe
    date_obj = parse_date_text(current_date_text)
    if date_obj is None:
        return []
    df = load_image_paths_for_date(date_obj, min_proba)
    df = filter_by_min_stations(df, min_stations)
    if df.empty:
        return []

    return images_for_group(df, group_ts)


def create_scrollable_burst_groups(df_all: pd.DataFrame) -> str:
    """
    Create HTML for scrollable burst groups with hover zoom functionality.
    Each entry's header format: "[timestamp] — Avg conf: [value]%"
    """
    if df_all.empty:
        return "<p>No burst data available.</p>"

    # Add CSS for hover zoom effect with fixed z-index and smart positioning
    css_style = """
    <style>
    .burst-image-container {
        position: relative;
        display: inline-block;
    }
    .hover-zoom {
        transition: transform 0.3s ease, z-index 0s;
        cursor: pointer;
        position: relative;
    }
    .hover-zoom:hover {
        transform: scale(2.5);
        z-index: 10000 !important;
        position: relative;
        border: 2px solid #007bff !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    /* First item in row - expand to the right */
    .burst-image-container:first-child .hover-zoom:hover {
        transform-origin: left center;
    }
    /* Last item in row - expand to the left */
    .burst-image-container:last-child .hover-zoom:hover {
        transform-origin: right center;
    }
    /* Middle items - expand from center */
    .burst-image-container:not(:first-child):not(:last-child) .hover-zoom:hover {
        transform-origin: center center;
    }
    </style>
    """

    # Group by TimeGroup and sort by newest first
    grouped = df_all.groupby("TimeGroup")
    html_content = css_style

    # Sort groups by newest first
    for group_time in sorted(grouped.groups.keys(), reverse=True):
        group_df = grouped.get_group(group_time).sort_values(
            by="Confidence", ascending=False
        )

        # Calculate stats for this group
        station_count = group_df["Instrument Location"].nunique()
        max_conf = group_df["Confidence"].max()
        avg_conf = group_df["Confidence"].mean()

        # Create header: datetime on top, avg conf below in smaller text
        burst_time = group_time.strftime("%Y-%m-%d %H:%M UTC")

        html_content += f"<div style='margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>"
        html_content += f"<h3 style='color: #333; margin-top: 0; margin-bottom: 2px;'>{burst_time}</h3>"
        html_content += f"<p style='color: #666; font-size: 0.9em; margin-top: 0; margin-bottom: 10px;'>Avg conf: {avg_conf:.1f}%</p>"

        # Group by instrument and show images
        html_content += "<div style='display: flex; flex-wrap: wrap; gap: 15px; position: relative; z-index: 1;'>"
        for instrument in group_df["Instrument Location"].unique():
            instrument_data = group_df[group_df["Instrument Location"] == instrument]
            best_detection = instrument_data.iloc[0]  # highest confidence

            # Use the correct Gradio API path for newer versions
            img_path: str = best_detection["Path"]

            html_content += f"<div class='burst-image-container' style='text-align: center; min-width: 200px;'>"
            html_content += f"<h4 style='margin: 5px 0; color: #666; font-size: 14px;'>{instrument}</h4>"
            html_content += f"<a href='/gradio_api/file={img_path}' target='_blank'>"
            html_content += f"<img src='/gradio_api/file={img_path}' class='hover-zoom' style='max-width: 180px; max-height: 180px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer;' alt='Burst detection' title='Click to open in new tab'>"
            html_content += f"</a>"
            html_content += f"<p style='margin: 5px 0; font-size: 12px; color: #888;'>Confidence: {best_detection['Confidence']:.1f}%</p>"
            html_content += "</div>"

        html_content += "</div></div>"

    return html_content


def load_latest(days_back: int, min_proba: float, min_stations: int):
    """
    Latest view: summarize groups across last N days, dropdown to pick a group, show gallery.
    We default to the newest group.
    """
    df_all = concatenate_days(days_back, min_proba)
    df_all = filter_by_min_stations(df_all, min_stations)
    if df_all.empty:
        msg = f"<div style='background:#fff3cd;padding:10px;border-radius:6px;'>No bursts in last {days_back} day(s).</div>"
        return (
            msg,
            pd.DataFrame(
                columns=["GroupStartUTC", "Stations", "Detections", "MaxConfidence"]
            ),
            gr.update(choices=[], value=None),
            [],
            [],
        )

    groups = summarize_groups(df_all)
    total = len(df_all)
    stations = df_all["Instrument Location"].nunique()
    msg = (
        f"<div style='background:#e7f3ff;padding:10px;border-radius:6px;'>"
        f"Last {days_back} day(s): {total} detections · {stations} station(s) · {len(groups)} group(s)</div>"
    )

    def label_row(r):
        t = pd.to_datetime(r["GroupStartUTC"])
        return f"{t.strftime('%Y-%m-%d %H:%M')}  |  {int(r['Stations'])} stations, {int(r['Detections'])} detections, max {r['MaxConfidence']:.1f}%"

    choices = [label_row(row) for _, row in groups.iterrows()]
    default_label = choices[0] if choices else None

    # Initial gallery = newest group
    gallery_items = []
    if default_label:
        ts = pd.to_datetime(default_label.split("|")[0].strip() + ":00")
        gallery_items = images_for_group(df_all, ts)

    return (
        msg,
        groups,
        gr.update(choices=choices, value=default_label),
        choices,
        gallery_items,
    )


# -----------------------------
# BUILD APP
# -----------------------------
def load_latest_bursts():
    """Load latest bursts with hardcoded settings: 3 days, conf ≥ 0.5, min 3 stations"""
    df_all = concatenate_days(days_back=3, min_proba=DEFAULT_MIN_PROBA)
    df_all = filter_by_min_stations(df_all, min_stations=DEFAULT_MIN_STATIONS)

    if df_all.empty:
        return "<p>No recent bursts found with the specified criteria.</p>"

    # Create scrollable burst groups with the SAME format that was working before
    html_content = create_scrollable_burst_groups(df_all)

    # Add explanatory note
    note = """
    <div style='margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.9em; color: #666;'>
        <em>Plots show last 3 days, with confidence ≥ 0.5 and at least 3 stations.</em>
    </div>
    """

    return html_content + note


def create_app():
    with gr.Blocks(
        title="FlareSense Burst Detection",
        css="""
        .sidebar { background-color: #f8f9fa; padding: 15px; border-radius: 8px; }
        .section-card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; background: #fff; }
        """,
    ) as demo:

        # HEADER
        gr.Markdown(
            f"""
            <div style="border:1px solid #ccc; padding:15px; border-radius:5px;">
              <h1 style="margin-top:0;">🌞 FlareSense by <a href="https://i4ds.ch/" target="_blank">i4ds@fhnw</a></h1>
              <p style="font-size:1.05em;">
                <b>Real-time detection of <a href="https://en.wikipedia.org/wiki/Solar_radio_burst" target="_blank">solar radio bursts</a> using <a href="https://www.e-callisto.org/" target="_blank">E-Callisto</a> data.</b><br></b><br>
                Updates every 15 minutes from <b>{len(INSTRUMENT_LIST)}</b> monitoring stations worldwide.<br>
                <small>Powered by <a href="https://huggingface.co/i4ds/flaresense-v2" target="_blank">FlareSense ML Model</a></small>
              </p>
            </div>
            """
        )

        with gr.Tabs():
            # -----------------------------------------------------
            # 1) LATEST (now FIRST tab)
            # -----------------------------------------------------
            with gr.TabItem("🔥 Latest"):
                gr.Markdown("### Recent Solar Radio Burst Detections")

                # Latest tab shows scrollable bursts with hardcoded settings
                latest_bursts_html = gr.HTML(
                    value="<div style='text-align: center; padding: 40px;'><p style='font-size: 1.2em;'>⏳ Loading latest bursts...</p></div>"
                )

                # Load latest bursts on page load with fixed settings
                demo.load(
                    fn=load_latest_bursts,
                    inputs=None,
                    outputs=latest_bursts_html,
                )

                # Refresh button for latest bursts
                refresh_latest_btn = gr.Button("🔄 Refresh Latest", variant="primary")
                refresh_latest_btn.click(
                    fn=load_latest_bursts,
                    inputs=None,
                    outputs=latest_bursts_html,
                )

            # -----------------------------------------------------
            # 2) TRENDS (load when tab is viewed)
            # -----------------------------------------------------
            with gr.TabItem("📈 Trends"):
                gr.Markdown("**Solar Radio Burst Trends (Last 60 Days)**")

                # Full width plot
                trend_plot = gr.Plot(show_label=False, container=True)
                statistics_html = gr.HTML()

                def load_all_trends():
                    return plot_all_data_with_ma(), generate_burst_statistics()

                # Load trends when the tab becomes visible
                demo.load(
                    fn=load_all_trends,
                    outputs=[trend_plot, statistics_html],
                )

            # -----------------------------------------------------
            # 3) DATA BROWSER & EXPORT (exactly like Latest but with date selection)
            # -----------------------------------------------------
            with gr.TabItem("🔎 Data Browser & Export"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### Date Range Selection", elem_classes=["section-card"]
                        )
                        date_start = gr.Textbox(
                            label="Start Date:",
                            value=today_date().strftime("%Y-%m-%d"),
                            placeholder="e.g. 2025-09-01, today, yesterday",
                        )
                        date_end = gr.Textbox(
                            label="End Date:",
                            value=today_date().strftime("%Y-%m-%d"),
                            placeholder="e.g. 2025-09-05, today",
                        )

                        gr.Markdown("### Filters", elem_classes=["section-card"])
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.1,
                            label="Minimum Confidence",
                        )
                        min_stations_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            label="Minimum Stations",
                        )

                        search_btn = gr.Button("🔍 Search", variant="primary")

                        gr.Markdown("### Export", elem_classes=["section-card"])
                        export_btn = gr.Button(
                            "📦 Download ZIP (Bursts + Images)", variant="secondary"
                        )
                        export_file = gr.File(
                            label="Download Export", interactive=False
                        )

                        gr.Markdown("### Links", elem_classes=["section-card"])
                        gr.Markdown(
                            """
                            - [E-Callisto Stations](https://www.e-callisto.org/stations.html)
                            - [Contact](mailto:vincenzo.timmel@fhnw.ch)
                            - [Research Paper](https://placeholder.link.to.paper)
                            """,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown(
                            "### Bursts for Selected Date Range\n"
                            "Shows bursts matching your filter criteria. ",
                            elem_classes=["section-card"],
                        )
                        date_bursts_html = gr.HTML(
                            value="<div style='text-align: center; padding: 40px;'><p style='font-size: 1.2em;'>⏳ Loading...</p></div>"
                        )

                def load_date_bursts(
                    start_date_val: str,
                    end_date_val: str,
                    min_conf: float,
                    min_stations: int,
                ):
                    """Load bursts for the selected date range with filters"""
                    dt_start = parse_date_text(start_date_val)
                    dt_end = parse_date_text(end_date_val)

                    if not dt_start:
                        return "<p>Invalid start date format. Please try again.</p>"
                    if not dt_end:
                        dt_end = dt_start  # Default to single day if end date invalid

                    # Ensure start <= end
                    if dt_start > dt_end:
                        dt_start, dt_end = dt_end, dt_start

                    # Limit range to 30 days max
                    max_days = 30
                    if (dt_end - dt_start).days > max_days:
                        return f"<p>Date range too large. Please select a range of {max_days} days or less.</p>"

                    # Load data for each day in range
                    all_dfs = []
                    current_date = dt_start
                    while current_date <= dt_end:
                        df_day = load_image_paths_for_date(
                            current_date, min_proba=min_conf
                        )
                        if not df_day.empty:
                            all_dfs.append(df_day)
                        current_date += timedelta(days=1)

                    if not all_dfs:
                        date_range_str = (
                            f"{dt_start.strftime('%Y-%m-%d')}"
                            if dt_start == dt_end
                            else f"{dt_start.strftime('%Y-%m-%d')} to {dt_end.strftime('%Y-%m-%d')}"
                        )
                        return f"<p>No bursts found for {date_range_str} with confidence ≥ {min_conf*100:.0f}%.</p>"

                    df_all = pd.concat(all_dfs, ignore_index=True)

                    # Apply minimum stations filter
                    df_all["TimeGroup"] = df_all["Datetime"].dt.floor("15min")
                    station_counts = (
                        df_all.groupby("TimeGroup")["Instrument Location"]
                        .nunique()
                        .reset_index()
                    )
                    station_counts.columns = ["TimeGroup", "station_count"]
                    valid_groups = station_counts[
                        station_counts["station_count"] >= min_stations
                    ]["TimeGroup"]
                    df_filtered = df_all[df_all["TimeGroup"].isin(valid_groups)]

                    if df_filtered.empty:
                        date_range_str = (
                            f"{dt_start.strftime('%Y-%m-%d')}"
                            if dt_start == dt_end
                            else f"{dt_start.strftime('%Y-%m-%d')} to {dt_end.strftime('%Y-%m-%d')}"
                        )
                        return f"<p>No bursts found for {date_range_str} with minimum {min_stations} station(s) and confidence ≥ {min_conf*100:.0f}%.</p>"

                    return create_scrollable_burst_groups(df_filtered)

                def export_date_data(
                    start_date_val: str,
                    end_date_val: str,
                    min_conf: float,
                    min_stations: int,
                ):
                    """Export ZIP with bursts and images for the selected date range"""
                    dt_start = parse_date_text(start_date_val)
                    dt_end = parse_date_text(end_date_val)

                    if not dt_start:
                        return None
                    if not dt_end:
                        dt_end = dt_start

                    # Ensure start <= end
                    if dt_start > dt_end:
                        dt_start, dt_end = dt_end, dt_start

                    # Limit range to 30 days
                    if (dt_end - dt_start).days > 30:
                        return None

                    # Load data for each day in range
                    all_dfs = []
                    current_date = dt_start
                    while current_date <= dt_end:
                        df_day = load_image_paths_for_date(
                            current_date, min_proba=min_conf
                        )
                        if not df_day.empty:
                            all_dfs.append(df_day)
                        current_date += timedelta(days=1)

                    if not all_dfs:
                        return None

                    df_all = pd.concat(all_dfs, ignore_index=True)
                    df_all = filter_by_min_stations(df_all, min_stations)

                    if df_all.empty:
                        return None

                    # Create ZIP export
                    tmp_dir = tempfile.mkdtemp()
                    date_str = (
                        f"{dt_start.strftime('%Y%m%d')}_to_{dt_end.strftime('%Y%m%d')}"
                        if dt_start != dt_end
                        else dt_start.strftime("%Y%m%d")
                    )
                    zip_path = os.path.join(
                        tmp_dir, f"FlareSense_Export_{date_str}.zip"
                    )

                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        # CSV (remove Path column)
                        csv_data = df_all.drop(columns=["TimeGroup", "Path"]).round(2)
                        csv_path = os.path.join(tmp_dir, f"bursts_{date_str}.csv")
                        csv_data.to_csv(csv_path, index=False)
                        zipf.write(csv_path, os.path.basename(csv_path))

                        # Images
                        for _, row in df_all.iterrows():
                            img_path = row["Path"]
                            if os.path.exists(img_path):
                                burst_time = row["Datetime"].strftime("%H-%M-%S")
                                burst_date = row["Datetime"].strftime("%Y-%m-%d")
                                confidence = row["Confidence"]
                                instrument = row["Instrument Location"]
                                filename = f"{burst_date}_{burst_time}_{instrument}_{confidence:.1f}pct.png"
                                zipf.write(img_path, f"images/{filename}")

                    return zip_path

                # Initial load
                demo.load(
                    fn=load_date_bursts,
                    inputs=[
                        date_start,
                        date_end,
                        confidence_slider,
                        min_stations_slider,
                    ],
                    outputs=[date_bursts_html],
                )

                # Search button click
                search_btn.click(
                    fn=load_date_bursts,
                    inputs=[
                        date_start,
                        date_end,
                        confidence_slider,
                        min_stations_slider,
                    ],
                    outputs=[date_bursts_html],
                )

                # Export functionality
                export_btn.click(
                    fn=export_date_data,
                    inputs=[
                        date_start,
                        date_end,
                        confidence_slider,
                        min_stations_slider,
                    ],
                    outputs=[export_file],
                )

        return demo


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import sys

    # Small guard: ensure regex util available
    # (we import inside main to avoid global import if not run)
    demo = create_app()
    demo.launch(allowed_paths=[BASE_PATH, "static"])
