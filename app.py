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
BASE_PATH = os.path.join("/mnt/nas05/data01/vincenzo/ecallisto/burst_live_images")
TIMEGROUP_MINUTES = 15
DEFAULT_MIN_PROBA = 0.5
DEFAULT_MIN_STATIONS = 1
DEFAULT_DAYS_BACK = 5
DAYS_SCROLL_MAX = 60  # how far back the "Days back" slider can go

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
        # CSV
        csv_data = table_data.drop(columns=["TimeGroup"]).round(2)
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


def plot_fast_daily_counts(days: int = 30):
    """
    Fast FS walk – counts images per station per day without parsing filenames.
    """
    base_date = today_date()
    daily_data = []

    for i in range(days):
        date = base_date - timedelta(days=i)
        year, month, day = ymd_from_date(date)
        day_path = os.path.join(BASE_PATH, year, month, day)
        if not os.path.exists(day_path):
            continue

        try:
            for station_dir in os.listdir(day_path):
                station_path = os.path.join(day_path, station_dir)
                if os.path.isdir(station_path) and station_dir in INSTRUMENT_LIST:
                    cnt = len(
                        [f for f in os.listdir(station_path) if f.endswith(".png")]
                    )
                    if cnt > 0:
                        daily_data.append(
                            {
                                "Date": f"{year}-{month}-{day}",
                                "Station": station_dir,
                                "Count": cnt,
                            }
                        )
        except (OSError, PermissionError):
            continue

    if not daily_data:
        return go.Figure()

    df = pd.DataFrame(daily_data)
    fig = px.bar(
        df,
        x="Date",
        y="Count",
        color="Station",
        title=f"Daily Burst Detections by Station (Last {days} Days)",
        labels={"Count": "Detections", "Date": "Date"},
        barmode="stack",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Detections",
        font=dict(size=14),
        xaxis={"categoryorder": "category ascending"},
    )
    return fig


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
    Create HTML for scrollable burst groups.
    Each entry's header format: "Burst at [timestamp] seen by X antennas — Max conf: [value], Avg conf: [value]"
    """
    if df_all.empty:
        return "<p>No burst data available.</p>"

    # Group by TimeGroup and sort by newest first
    grouped = df_all.groupby("TimeGroup")
    html_content = ""

    # Sort groups by newest first
    for group_time in sorted(grouped.groups.keys(), reverse=True):
        group_df = grouped.get_group(group_time).sort_values(
            by="Confidence", ascending=False
        )

        # Calculate stats for this group
        station_count = group_df["Instrument Location"].nunique()
        max_conf = group_df["Confidence"].max()
        avg_conf = group_df["Confidence"].mean()

        # Create header with requested format
        burst_time = group_time.strftime("%Y-%m-%d %H:%M UTC")
        header = f"Burst at {burst_time} seen by {station_count} antennas — Max conf: {max_conf:.1f}%, Avg conf: {avg_conf:.1f}%"

        html_content += f"<div style='margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>"
        html_content += f"<h3 style='color: #333; margin-top: 0;'>{header}</h3>"

        # Group by instrument and show images
        html_content += "<div style='display: flex; flex-wrap: wrap; gap: 15px;'>"
        for instrument in group_df["Instrument Location"].unique():
            instrument_data = group_df[group_df["Instrument Location"] == instrument]
            best_detection = instrument_data.iloc[0]  # highest confidence

            # Use the correct Gradio API path for newer versions
            img_path = best_detection["Path"]

            html_content += f"<div style='text-align: center; min-width: 200px;'>"
            html_content += f"<h4 style='margin: 5px 0; color: #666; font-size: 14px;'>{instrument}</h4>"
            html_content += f"<img src='/gradio_api/file={img_path}' style='max-width: 180px; max-height: 180px; border: 1px solid #ccc; border-radius: 4px;' alt='Burst detection'>"
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
                <b>Real-time detection of solar radio bursts using <a href="https://www.e-callisto.org/" target="_blank">E-Callisto</a> data.</b><br>
                Updates every 30 minutes from <b>{len(INSTRUMENT_LIST)}</b> monitoring stations worldwide.
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
                gr.Markdown("**Scrollable burst groups from the last 3 days**")

                # Latest tab shows scrollable bursts with hardcoded settings
                latest_bursts_html = gr.HTML()

                # Load latest bursts on page load with fixed settings
                def load_latest_bursts():
                    """Load latest bursts with hardcoded settings: 3 days, conf ≥ 0.5, min 3 stations"""
                    df_all = concatenate_days(days_back=3, min_proba=0.5)
                    df_all = filter_by_min_stations(df_all, min_stations=3)

                    if df_all.empty:
                        return (
                            "<p>No recent bursts found with the specified criteria.</p>"
                        )

                    # Create scrollable burst groups with the SAME format that was working before
                    html_content = create_scrollable_burst_groups(df_all)

                    # Add explanatory note
                    note = """
                    <div style='margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.9em; color: #666;'>
                        <em>Plots show last 3 days, with confidence ≥ 0.5 and at least 3 stations.</em>
                    </div>
                    """

                    return html_content + note

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
            # 2) TRENDS (now SECOND tab)
            # -----------------------------------------------------
            with gr.TabItem("📈 Trends"):
                with gr.Row():
                    with gr.Column(scale=1):
                        trend_days = gr.Slider(
                            7, 90, value=30, step=1, label="Days span"
                        )
                        trend_min_proba = gr.Slider(
                            0.1,
                            1.0,
                            value=DEFAULT_MIN_PROBA,
                            step=0.1,
                            label="Min Confidence",
                        )
                        refresh_trends = gr.Button("🔄 Refresh", variant="primary")
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "**Bursts per day by station (stacked) with 7‑day moving average**"
                        )
                        trend_plot = gr.Plot()
                        gr.Markdown(
                            "**Fast file-based detection counts (sanity check)**"
                        )
                        fast_plot = gr.Plot()

                def load_trends(days: int, min_p: float):
                    return plot_daily_by_station(days, min_p), plot_fast_daily_counts(
                        days
                    )

                demo.load(
                    fn=load_trends,
                    inputs=[trend_days, trend_min_proba],
                    outputs=[trend_plot, fast_plot],
                )
                refresh_trends.click(
                    fn=load_trends,
                    inputs=[trend_days, trend_min_proba],
                    outputs=[trend_plot, fast_plot],
                )

            # -----------------------------------------------------
            # 3) DATA BROWSER & EXPORT (now THIRD tab)
            # -----------------------------------------------------
            with gr.TabItem("🔎 Data Browser & Export"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Controls", elem_classes=["section-card"])
                        date_text = gr.Textbox(
                            label="Date (free text):",
                            value=today_date().strftime("%Y-%m-%d"),
                            placeholder="e.g. 2025-09-01, 9.1.2025, today, yesterday",
                        )
                        days_back_slider = gr.Slider(
                            minimum=0,
                            maximum=DAYS_SCROLL_MAX,
                            step=1,
                            value=0,
                            label="Days back (0 = today)",
                            info="Slide to quickly browse days",
                        )
                        min_proba = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=DEFAULT_MIN_PROBA,
                            step=0.1,
                            label="Min Confidence",
                            info="Minimum confidence threshold (probability)",
                        )
                        min_stations = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=DEFAULT_MIN_STATIONS,
                            step=1,
                            label="Min Stations",
                            info="Burst must be detected by at least this many stations",
                        )
                        with gr.Row():
                            prev_day_btn = gr.Button(
                                "⬅️ Previous Day", variant="secondary"
                            )
                            next_day_btn = gr.Button("Next Day ➡️", variant="secondary")
                        search_btn = gr.Button(
                            "🔍 Browse Selected Day", variant="primary"
                        )

                        gr.Markdown("### Export", elem_classes=["section-card"])
                        export_btn = gr.Button(
                            "📦 Export ZIP (CSV + Images)", variant="secondary"
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
                        gr.Markdown("### Summary", elem_classes=["section-card"])
                        day_summary_html = gr.HTML()
                        gr.Markdown("### Burst Groups", elem_classes=["section-card"])
                        groups_table = gr.Dataframe(
                            headers=[
                                "GroupStartUTC",
                                "Stations",
                                "Detections",
                                "MaxConfidence",
                            ],
                            datatype=[
                                "str",
                                "number",
                                "number",
                                "number",
                            ],  # Changed datetime to str
                            interactive=False,
                            row_count=5,
                        )
                        group_selector = gr.Dropdown(
                            label="Select burst group",
                            choices=[],
                        )
                        # Hidden mapping (list of ISO strings aligned with dropdown choices)
                        hidden_iso_list = gr.State([])

                        gr.Markdown("### Group Images", elem_classes=["section-card"])
                        gallery = gr.Gallery(
                            label="Detections in selected group",
                            show_label=True,
                            allow_preview=True,
                            columns=3,
                            height="auto",
                        )

                # Wiring — initial load (today from slider = 0)
                demo.load(
                    fn=browser_day_from_slider,
                    inputs=[days_back_slider, date_text, min_proba, min_stations],
                    outputs=[
                        date_text,
                        day_summary_html,
                        groups_table,
                        group_selector,
                        hidden_iso_list,
                        gallery,
                    ],
                )

                # Search by date text
                search_btn.click(
                    fn=browser_day_from_text,
                    inputs=[date_text, days_back_slider, min_proba, min_stations],
                    outputs=[
                        date_text,
                        day_summary_html,
                        groups_table,
                        group_selector,
                        hidden_iso_list,
                        gallery,
                    ],
                )

                # Navigate by slider (scroll days)
                days_back_slider.change(
                    fn=browser_day_from_slider,
                    inputs=[days_back_slider, date_text, min_proba, min_stations],
                    outputs=[
                        date_text,
                        day_summary_html,
                        groups_table,
                        group_selector,
                        hidden_iso_list,
                        gallery,
                    ],
                )

                # Prev/Next day buttons mutate the slider
                def dec_day(val: int):
                    return max(0, val + 1)

                def inc_day(val: int):
                    return max(0, val - 1)

                prev_day_btn.click(
                    fn=dec_day, inputs=[days_back_slider], outputs=[days_back_slider]
                )
                next_day_btn.click(
                    fn=inc_day, inputs=[days_back_slider], outputs=[days_back_slider]
                )

                # Update gallery when a group is selected
                group_selector.change(
                    fn=update_gallery_for_selected_group,
                    inputs=[
                        group_selector,
                        hidden_iso_list,
                        date_text,
                        min_proba,
                        min_stations,
                    ],
                    outputs=[gallery],
                )

                # Export ZIP (for current date / filters)
                def do_export(date_text_val: str, min_p: float, min_k: int):
                    dt = parse_date_text(date_text_val)
                    if not dt:
                        return None
                    return create_zip_export_for_date(dt, min_p, min_k)

                export_btn.click(
                    fn=do_export,
                    inputs=[date_text, min_proba, min_stations],
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
    demo.launch(allowed_paths=[BASE_PATH])
