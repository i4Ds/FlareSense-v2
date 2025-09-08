import gradio as gr
import glob, os
from datetime import datetime, timedelta
import os, glob
import tempfile
import pandas as pd
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import shutil


BASE_PATH = os.path.join("/mnt/nas05/data01/vincenzo/ecallisto/burst_live_images")
SORT_BY_COLUMN = "Datetime"
MIN_PROBABILITY = 0.5

from pred_live import INSTRUMENT_LIST


def load_images(table, sort_by) -> List:
    if sort_by == "Confidence":
        table = table.sort_values(by="Confidence", ascending=False)
    else:
        table = table.sort_values(by="Datetime", ascending=False)

    img_data = []
    for _, row in table.iterrows():
        img_data.append((row["Path"], row["Confidence"]))

    if len(img_data) == 0:
        return ["style/DALLE_ERROR.png"]

    return [(x[0], f"Confidence: {x[1]:.2f} %") for x in img_data]


def load_image_paths(year, month, day, min_proba) -> pd.DataFrame:
    table_data = []
    search_path = os.path.join(BASE_PATH, year, month, day, "*", "*")
    for f in glob.glob(search_path):
        base = os.path.basename(f)
        parts = base.split("_")
        proba = float(parts[0])
        if proba < min_proba:
            continue
        antenna = "_".join(parts[1:-2])
        dt_str = parts[-2] + " " + parts[-1].replace(".png", "")
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")
        table_data.append([dt, antenna, proba, f])
    df = pd.DataFrame(
        table_data, columns=["Datetime", "Instrument Location", "Confidence", "Path"]
    ).sort_values(by="Datetime", ascending=True)

    # Filter out antennas not in INSTRUMENT_LIST
    df = df[df["Instrument Location"].isin(INSTRUMENT_LIST)]

    # Add TimeGroup for 15-minute intervals
    df["TimeGroup"] = df["Datetime"].dt.floor("15min")

    return df


def download_csv(year, month, day, sort_by, min_proba, k) -> str:
    # Return CSV for immediate download
    _, df = load_images_and_table(year, month, day, sort_by, min_proba, k)
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, f"FlareSense_BurstPlots_{year}_{month}_{day}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def create_zip_export(year, month, day, min_proba, k) -> str:
    """Create a ZIP file with CSV and all burst images for the selected date."""
    table_data = load_image_paths(year, month, day, min_proba)

    if table_data.empty:
        return None

    # Filter by minimum stations - same logic as load_images_and_table
    station_counts = table_data.groupby("TimeGroup")["Instrument Location"].nunique()
    valid_time_groups = station_counts[station_counts >= k].index
    table_data = table_data[table_data["TimeGroup"].isin(valid_time_groups)]

    if table_data.empty:
        return None

    # Create temporary directory
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, f"FlareSense_Export_{year}_{month}_{day}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add CSV file
        csv_data = table_data.drop(columns=["TimeGroup"]).round(2)
        csv_path = os.path.join(tmp_dir, f"bursts_{year}_{month}_{day}.csv")
        csv_data.to_csv(csv_path, index=False)
        zipf.write(csv_path, f"bursts_{year}_{month}_{day}.csv")

        # Add all image files
        for _, row in table_data.iterrows():
            img_path = row["Path"]
            if os.path.exists(img_path):
                # Create a meaningful filename for the ZIP
                burst_time = row["Datetime"].strftime("%H-%M-%S")
                burst_date = row["Datetime"].strftime("%Y-%m-%d")
                confidence = row["Confidence"]
                instrument = row["Instrument Location"]
                filename = (
                    f"{burst_date}_{burst_time}_{instrument}_{confidence:.1f}pct.png"
                )
                zipf.write(img_path, f"images/{filename}")

    return zip_path


def get_grouped_bursts_display(table_data: pd.DataFrame) -> str:
    """Create HTML display for grouped bursts (shared between latest and advanced tools)."""
    if table_data.empty:
        return "<p>No bursts found.</p>"

    # Group by TimeGroup
    grouped = table_data.groupby("TimeGroup")
    html_content = ""

    # Sort groups by newest first
    for group_time in sorted(grouped.groups.keys(), reverse=True):
        group_df = grouped.get_group(group_time).sort_values(
            by="Confidence", ascending=False
        )

        # Create title for this burst group
        burst_time = group_time.strftime("%Y-%m-%d %H:%M UTC")
        station_count = group_df["Instrument Location"].nunique()
        html_content += f"<div style='margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>"
        html_content += f"<h3 style='color: #333; margin-top: 0;'>🌞 Burst at {burst_time} <span style='color: #666; font-size: 0.8em;'>({station_count} stations)</span></h3>"

        # Group by instrument and show images
        html_content += "<div style='display: flex; flex-wrap: wrap; gap: 15px;'>"
        for instrument in group_df["Instrument Location"].unique():
            instrument_data = group_df[group_df["Instrument Location"] == instrument]
            best_detection = instrument_data.iloc[0]  # highest confidence

            # Use proper Gradio file serving with absolute path
            img_path = best_detection["Path"]

            html_content += f"<div style='text-align: center; min-width: 200px;'>"
            html_content += f"<h4 style='margin: 5px 0; color: #666; font-size: 14px;'>{instrument}</h4>"
            html_content += f"<img src='file={img_path}' style='max-width: 180px; max-height: 180px; border: 1px solid #ccc; border-radius: 4px;' alt='Burst detection'>"
            html_content += f"<p style='margin: 5px 0; font-size: 12px; color: #888;'>Confidence: {best_detection['Confidence']:.1f}%</p>"
            html_content += "</div>"

        html_content += "</div></div>"

    return html_content


def plot_bursts(min_proba=0.5, days=30):
    try:
        dfs = []
        for i in range(days):
            d = datetime.now() - timedelta(days=i)
            y, m, da = str(d.year), f"{d.month:02d}", f"{d.day:02d}"
            df = load_image_paths(y, m, da, min_proba)
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return {}

        all_data = pd.concat(dfs, ignore_index=True)

        all_data["Day"] = all_data["Datetime"].dt.floor("D")
        # Calculate total daily counts
        daily_counts = all_data.groupby("Day").size().reset_index(name="Total Count")
        daily_counts["7-Day MA"] = daily_counts["Total Count"].rolling(window=7).mean()

        # Group by day and station, count
        daily_by_station = (
            all_data.groupby(["Day", "Instrument Location"])
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
        )

        # Stacked bar plot
        fig = px.bar(
            daily_by_station,
            x="Day",
            y="Count",
            color="Instrument Location",
            barmode="stack",
            title=f"Number of Bursts per Day by Station (Last {days} Days)",
            labels={"Day": "Date", "Count": "Bursts"},
        )
        # Add moving average line to the plot
        fig.add_scatter(
            x=daily_counts["Day"],
            y=daily_counts["7-Day MA"],
            mode="lines",
            line=dict(color="red", width=2),
            name="7-Day Moving Average",
        )

        fig.update_layout(
            hovermode="closest",
            xaxis_title="Date",
            yaxis_title="Bursts Detected",
            font=dict(size=14),
        )
        # Show only hovered bar info
        fig.update_traces(hovertemplate="Date: %{x}<br>Bursts: %{y}")

        return fig
    except Exception as e:
        print(f"Error in plot_bursts: {e}")
        return {}


def load_images_and_table(
    year, month, day, sort_by, min_proba, k
) -> Tuple[List, pd.DataFrame]:
    table_data = load_image_paths(year, month, day, min_proba)
    # Filter data by minimum number of stations
    filtered_table = (
        table_data.drop(columns=["Path"])
        .sort_values(by=["Datetime", "Instrument Location"], ascending=[True, True])
        .groupby(["Datetime"])
        .agg(
            {
                "Datetime": "first",
                "Instrument Location": lambda x: ", ".join(set(x)),
                "Confidence": "mean",
            }
        )
        .reset_index(drop=True)
    ).copy()
    # Calculate the number of unique stations per burst
    filtered_table["count"] = filtered_table["Instrument Location"].apply(
        lambda x: len(x.split(","))
    )

    # Filter bursts detected by at least k stations
    filtered_table: pd.DataFrame = filtered_table[filtered_table["count"] >= k].drop(
        columns=["count"]
    )

    # Filter table data
    table_data = table_data[table_data["Datetime"].isin(filtered_table["Datetime"])]

    # Group by TimeGroup and collect images with group titles
    grouped = table_data.groupby("TimeGroup")
    img_data = []
    for group_time in sorted(grouped.groups.keys(), reverse=True):  # Newest first
        group_df = grouped.get_group(group_time).sort_values(
            by="Confidence", ascending=False
        )
        for _, row in group_df.iterrows():
            label = f"Burst at {group_time.strftime('%Y-%m-%d %H:%M')} - Confidence: {row['Confidence']:.2f} %"
            img_data.append((row["Path"], label))

    if len(img_data) == 0:
        img_data = ["style/DALLE_ERROR.png"]

    return img_data, filtered_table.round(2)


def get_current_date():
    current_datetime = datetime.now()
    current_year = str(current_datetime.year)
    current_month = f"{current_datetime.month:02d}"
    current_day = f"{current_datetime.day:02d}"
    return current_year, current_month, current_day


def get_latest_bursts(
    days_back=5, min_proba=0.5, min_stations=1
) -> str:  # Increased days_back from 2 to 5, changed default min_stations from 3 to 1
    """Get the latest bursts and format them as HTML."""
    dfs = []
    for i in range(days_back):
        d = datetime.now() - timedelta(days=i)
        y, m, da = str(d.year), f"{d.month:02d}", f"{d.day:02d}"
        df = load_image_paths(y, m, da, min_proba)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return "<p>No recent bursts found.</p>"

    all_data = pd.concat(dfs, ignore_index=True)

    # Filter by minimum stations - group by TimeGroup and count unique stations
    station_counts = all_data.groupby("TimeGroup")["Instrument Location"].nunique()
    valid_time_groups = station_counts[station_counts >= min_stations].index
    all_data = all_data[all_data["TimeGroup"].isin(valid_time_groups)]

    if all_data.empty:
        return f"<p>No bursts found that were detected by at least {min_stations} stations.</p>"

    return get_grouped_bursts_display(all_data)


def get_advanced_bursts(year, month, day, min_proba, min_stations) -> str:
    """Get bursts for specific date with grouping for advanced tools."""
    table_data = load_image_paths(year, month, day, min_proba)

    if table_data.empty:
        return "<p>No bursts found for this date.</p>"

    # Filter by minimum stations
    station_counts = table_data.groupby("TimeGroup")["Instrument Location"].nunique()
    valid_time_groups = station_counts[station_counts >= min_stations].index
    table_data = table_data[table_data["TimeGroup"].isin(valid_time_groups)]

    if table_data.empty:
        return f"<p>No bursts found that were detected by at least {min_stations} stations on {year}-{month}-{day}.</p>"

    return get_grouped_bursts_display(table_data)


def get_all_bursts_data(min_proba=0.5) -> pd.DataFrame:
    """Get all available burst data using glob search."""
    all_files = glob.glob(os.path.join(BASE_PATH, "*", "*", "*", "*", "*.png"))
    table_data = []

    for f in all_files:
        try:
            base = os.path.basename(f)
            parts = base.split("_")
            proba = float(parts[0])
            if proba < min_proba:
                continue
            antenna = "_".join(parts[1:-2])
            dt_str = parts[-2] + " " + parts[-1].replace(".png", "")
            dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")
            table_data.append([dt, antenna, proba, f])
        except (ValueError, IndexError):
            continue  # Skip malformed filenames

    if not table_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        table_data, columns=["Datetime", "Instrument Location", "Confidence", "Path"]
    ).sort_values(by="Datetime", ascending=True)

    # Filter out antennas not in INSTRUMENT_LIST
    df = df[df["Instrument Location"].isin(INSTRUMENT_LIST)]

    # Add TimeGroup for 15-minute intervals
    df["TimeGroup"] = df["Datetime"].dt.floor("15min")

    return df


def create_interactive_data_browser(
    year="2025", month="01", day="01", min_proba=0.5
) -> str:
    """Create an interactive, scrollable data browser with actual burst images."""
    df = load_image_paths(year, month, day, min_proba)

    if df is None or df.empty:
        return f"""
        <div style='text-align: center; padding: 40px; color: #666;'>
            <h3>📅 No bursts found for {day}/{month}/{year}</h3>
            <p>Try a different date or lower the confidence threshold.</p>
        </div>
        """

    # Group by TimeGroup and sort by time
    grouped = (
        df.groupby("TimeGroup")
        .apply(lambda x: x.sort_values("Confidence", ascending=False))
        .reset_index(drop=True)
    )
    time_groups = grouped.groupby("TimeGroup")

    html_parts = [
        f"""
    <div class='interactive-browser'>
        <div style='padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px 8px 0 0;'>
            <h3 style='margin: 0; color: white;'>🔍 Burst Explorer: {day}/{month}/{year}</h3>
        </div>
        <div style='padding: 15px; background: #e3f2fd; border-bottom: 1px solid #ddd;'>
            <strong>📊 Summary:</strong> {len(df)} detections across {len(time_groups)} time groups
        </div>
        <div style='padding: 15px;'>
    """
    ]

    for time_group, group_data in time_groups:
        # Count stations for this time group
        stations = group_data["Instrument Location"].nunique()
        max_conf = group_data["Confidence"].max()

        time_str = time_group.strftime("%H:%M")

        html_parts.append(
            f"""
        <div class='burst-group' style='border: 1px solid #ddd; margin-bottom: 15px; border-radius: 12px; background: white; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <div style='background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); padding: 12px; border-bottom: 1px solid #ddd;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: bold; color: #495057;'>🕐 {time_str}</span>
                    <div style='display: flex; gap: 15px; font-size: 14px; color: #6c757d;'>
                        <span>📡 {stations} station(s)</span>
                        <span>🎯 Max: {max_conf:.1f}%</span>
                    </div>
                </div>
            </div>
            <div style='padding: 20px;'>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;'>
        """
        )

        # Add images for this time group
        for _, row in group_data.iterrows():
            station = row["Instrument Location"]
            confidence = row["Confidence"]
            img_path = row["Path"]

            # Use proper Gradio file serving with absolute path

            html_parts.append(
                f"""
                <div style='border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); transition: all 0.3s ease;' onmouseover='this.style.transform="translateY(-3px)"; this.style.boxShadow="0 6px 20px rgba(0,0,0,0.15)"' onmouseout='this.style.transform="translateY(0px)"; this.style.boxShadow="0 2px 8px rgba(0,0,0,0.1)"'>
                    <div style='font-weight: bold; margin-bottom: 8px; color: #495057; font-size: 14px;'>{station}</div>
                    <div style='color: #28a745; margin-bottom: 12px; font-size: 13px; font-weight: 600;'>Confidence: {confidence:.1f}%</div>
                    <img class='burst-image' src='file={img_path}' style='max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                </div>
            """
            )

        html_parts.append(
            """
                </div>
            </div>
        </div>
        """
        )

    html_parts.append(
        """
        </div>
    </div>
    """
    )

    return "".join(html_parts)


def get_fast_daily_counts(days=30) -> pd.DataFrame:
    """Fast count of burst files per day by counting directory structure."""
    daily_data = []
    base_date = datetime.now()

    for i in range(days):
        date = base_date - timedelta(days=i)
        year, month, day = date.year, f"{date.month:02d}", f"{date.day:02d}"

        # Count files for this day by iterating through station directories
        day_path = os.path.join(BASE_PATH, str(year), month, day)
        if not os.path.exists(day_path):
            continue

        total_files = 0
        station_counts = {}

        # Get all station directories for this day
        try:
            for station_dir in os.listdir(day_path):
                station_path = os.path.join(day_path, station_dir)
                if os.path.isdir(station_path) and station_dir in INSTRUMENT_LIST:
                    # Count PNG files in this station directory
                    png_files = len(
                        [f for f in os.listdir(station_path) if f.endswith(".png")]
                    )
                    if png_files > 0:
                        station_counts[station_dir] = png_files
                        total_files += png_files
        except (OSError, PermissionError):
            continue

        if total_files > 0:
            for station, count in station_counts.items():
                daily_data.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Station": station,
                        "Count": count,
                    }
                )

    if not daily_data:
        return pd.DataFrame()

    return pd.DataFrame(daily_data)


def create_fast_daily_plot(days=30):
    """Create a fast daily burst count plot."""
    try:
        df = get_fast_daily_counts(days)

        if df.empty:
            return {}

        # Create stacked bar chart
        fig = px.bar(
            df,
            x="Date",
            y="Count",
            color="Station",
            title=f"Daily Burst Detections by Station (Last {days} Days)",
            labels={"Count": "Number of Detections", "Date": "Date"},
            barmode="stack",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Detections",
            font=dict(size=14),
            xaxis={"categoryorder": "category ascending"},
        )

        return fig
    except Exception as e:
        print(f"Error creating fast daily plot: {e}")
        return {}


def create_simple_data_browser() -> Tuple[str, any]:
    """Create a simpler data browser with fast daily plot."""
    try:
        # Get fast daily plot
        fig = create_fast_daily_plot(30)

        # Create summary text with basic stats
        total_days_with_data = 0
        if fig:  # Check if we have data
            daily_counts = get_fast_daily_counts(30)
            if not daily_counts.empty:
                total_detections = daily_counts["Count"].sum()
                unique_stations = daily_counts["Station"].nunique()
                total_days_with_data = daily_counts["Date"].nunique()
                date_range = (
                    f"{daily_counts['Date'].min()} to {daily_counts['Date'].max()}"
                )

                html_content = f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <h3>📊 Quick Database Overview (Last 30 Days)</h3>
                <ul>
                    <li><strong>Total Detections:</strong> {total_detections:,}</li>
                    <li><strong>Active Days:</strong> {total_days_with_data}</li>
                    <li><strong>Active Stations:</strong> {unique_stations}</li>
                    <li><strong>Date Range:</strong> {date_range}</li>
                </ul>
                <p><em>Fast count based on file system structure. Use date selector above for detailed exploration.</em></p>
                </div>
                """
            else:
                html_content = "<p>No recent burst data found.</p>"
        else:
            html_content = "<p>No recent burst data found.</p>"

        return html_content, fig if fig else {}

    except Exception as e:
        error_html = f"""
        <div style='background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24;'>
        <h3>⚠️ Error Loading Data</h3>
        <p>Could not load database overview: {str(e)}</p>
        </div>
        """
        return error_html, {}


def get_data_browser_content(min_proba=0.5) -> Tuple[str, any]:
    """Get content for data browser page."""
    return create_simple_data_browser()


def create_demo():
    years = ["2025", "2024"]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    current_year, current_month, current_day = get_current_date()

    # Create the Gradio interface
    with gr.Blocks(
        title="FlareSense Burst Detection",
        css="""
        .burst-container { margin-bottom: 20px; }
        .sidebar { background-color: #f8f9fa; padding: 15px; border-radius: 8px; }
        .interactive-browser { 
            max-height: 600px; 
            overflow-y: auto; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            background: #fafafa;
        }
        .burst-group { 
            transition: all 0.3s ease; 
            margin-bottom: 10px;
        }
        .burst-group:hover { 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
            transform: translateY(-2px);
        }
        .burst-image {
            transition: transform 0.3s ease;
        }
        .burst-image:hover {
            transform: scale(1.05);
        }
    """,
    ) as demo:
        gr.Markdown(
            f"""
            <div style="border:1px solid #ccc; padding:15px; border-radius:5px;">
            <h1 style="margin-top:0;">🌞 FlareSense by <a href="https://i4ds.ch/" target="_blank">i4ds@fhnw</a></h1>
            <p style="font-size:1.1em;">
            <b>Real-time detection of solar radio bursts using <a href="https://www.e-callisto.org/" target="_blank">E-Callisto</a> data.</b><br>
            Updates every 30 minutes from <b>{len(INSTRUMENT_LIST)}</b> monitoring stations worldwide.
            </p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("🔥 Latest Bursts"):
                with gr.Row():
                    with gr.Column(scale=4):  # Main content area
                        gr.Markdown("## Recent Solar Radio Burst Detections")

                        # Controls for filtering
                        with gr.Row():
                            min_stations_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=1,  # Changed from 3 to 1 to show more bursts
                                step=1,
                                label="Minimum Stations Required",
                                info="Bursts must be detected by at least this many stations",
                            )
                            refresh_btn = gr.Button(
                                "🔄 Refresh", variant="primary", scale=0
                            )

                        bursts_html = gr.HTML()

                        # Function to load bursts with station filter
                        def load_latest_with_filter(min_stations):
                            return get_latest_bursts(
                                days_back=5,
                                min_proba=0.5,
                                min_stations=min_stations,  # Updated days_back
                            )

                        # Load latest bursts on page load
                        demo.load(
                            fn=lambda: load_latest_with_filter(
                                1
                            ),  # Changed from 3 to 1
                            inputs=None,
                            outputs=bursts_html,
                        )

                        # Refresh button and slider updates
                        refresh_btn.click(
                            fn=load_latest_with_filter,
                            inputs=min_stations_slider,
                            outputs=bursts_html,
                        )

                        min_stations_slider.change(
                            fn=load_latest_with_filter,
                            inputs=min_stations_slider,
                            outputs=bursts_html,
                        )

                    with gr.Column(scale=1):  # Sidebar
                        gr.Markdown("### ℹ️ Information", elem_classes=["sidebar"])
                        gr.Markdown(
                            """
                        **Links:**
                        - [E-Callisto Stations](https://www.e-callisto.org/stations.html)
                        - [Contact Us](mailto:vincenzo.timmel@fhnw.ch)
                        - [Research Paper](https://placeholder.link.to.paper)
                        
                        **Update Frequency:** Every 30 minutes
                        
                        **About:** This page shows solar radio burst detections from today and yesterday. Use the slider to filter bursts by the minimum number of detecting stations to reduce false positives.
                        """
                        )

            with gr.TabItem("📊 Data Browser"):
                gr.Markdown("## Interactive Burst Explorer")
                gr.Markdown(
                    "Browse and scroll through actual burst detections. Select a date to explore."
                )

                # Date selection for data browser
                with gr.Row():
                    with gr.Column(scale=1):
                        browser_year = gr.Dropdown(
                            choices=years, label="Year", value=current_year
                        )
                        browser_month = gr.Dropdown(
                            choices=months, label="Month", value=current_month
                        )
                        browser_day = gr.Dropdown(
                            choices=days, label="Day", value=current_day
                        )
                        browser_min_proba = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Min Confidence",
                            info="Minimum confidence threshold",
                        )
                        browse_btn = gr.Button("🔍 Browse Bursts", variant="primary")

                    with gr.Column(scale=2):
                        # Summary info
                        browser_summary_html = gr.HTML(
                            value="<p>Select a date and click 'Browse Bursts' to explore detections.</p>"
                        )

                # Interactive burst display
                interactive_browser_html = gr.HTML()

                # Browse button functionality
                def browse_date_bursts(year, month, day, min_proba):
                    # Get summary for the selected date
                    df = load_image_paths(year, month, day, min_proba)
                    if df is None or df.empty:
                        summary = f"<div style='background: #fff3cd; padding: 10px; border-radius: 5px;'>No bursts found for {day}/{month}/{year} with confidence ≥ {min_proba:.1f}</div>"
                        interactive_html = ""
                    else:
                        total_detections = len(df)
                        unique_stations = df["Instrument Location"].nunique()
                        time_groups = df["TimeGroup"].nunique()
                        summary = f"""
                        <div style='background: #d4edda; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>📅 {day}/{month}/{year}:</strong> {total_detections} detections, {unique_stations} stations, {time_groups} time groups
                        </div>
                        """
                        interactive_html = create_interactive_data_browser(
                            year, month, day, min_proba
                        )

                    return summary, interactive_html

                browse_btn.click(
                    fn=browse_date_bursts,
                    inputs=[
                        browser_year,
                        browser_month,
                        browser_day,
                        browser_min_proba,
                    ],
                    outputs=[browser_summary_html, interactive_browser_html],
                )

                # Statistics plot and summary (moved to bottom)
                with gr.Accordion("📈 Database Statistics", open=False):
                    gr.Markdown("Overall database statistics and trends.")
                    stats_plot = gr.Plot()
                    stats_summary_html = gr.HTML()

                    # Load initial database stats
                    def load_database_stats():
                        html_content, plot_fig = get_data_browser_content()
                        return html_content, plot_fig

                    demo.load(
                        fn=load_database_stats,
                        inputs=None,
                        outputs=[stats_summary_html, stats_plot],
                    )

                    refresh_stats_btn = gr.Button(
                        "🔄 Refresh Statistics", variant="secondary"
                    )
                    refresh_stats_btn.click(
                        fn=load_database_stats,
                        inputs=None,
                        outputs=[stats_summary_html, stats_plot],
                    )

            with gr.TabItem("🔧 Advanced Tools"):
                gr.Markdown("## Advanced Search & Export Tools")

                with gr.Row():
                    with gr.Column(scale=2):  # Larger column for search
                        gr.Markdown("### Specific Date Search")
                        with gr.Row():
                            year = gr.Dropdown(
                                choices=years, label="Year", value=current_year
                            )
                            month = gr.Dropdown(
                                choices=months, label="Month", value=current_month
                            )
                            day = gr.Dropdown(
                                choices=days, label="Day", value=current_day
                            )
                            min_stations_adv = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Min Stations",
                                info="Minimum detecting stations",
                            )

                        load_grouped_btn = gr.Button(
                            "🔍 Load Grouped Bursts", variant="primary"
                        )
                        advanced_bursts_html = gr.HTML()

                    with gr.Column(scale=1):  # Smaller column for export
                        gr.Markdown("### Export Data")
                        gr.Markdown(
                            "Create a ZIP file with CSV data and all burst images for the selected date."
                        )
                        export_zip_btn = gr.Button(
                            "📦 Export ZIP with Images", variant="secondary"
                        )
                        export_file = gr.File(label="Download Export")

                        gr.Markdown("---")
                        gr.Markdown("### Legacy Gallery View")
                        legacy_btn = gr.Button("� Show Gallery", size="sm")
                        gallery = gr.Gallery(
                            object_fit="fill",
                            elem_id="gallery",
                            columns=[2],
                            rows=[1],
                            height="200px",
                            visible=False,
                        )

        # Hidden states for advanced search
        sort_by = gr.State("Datetime")
        min_proba = gr.State(0.5)

        # Advanced search functionality - grouped bursts
        load_grouped_btn.click(
            fn=get_advanced_bursts,
            inputs=[year, month, day, min_proba, min_stations_adv],
            outputs=advanced_bursts_html,
        )

        # ZIP export functionality
        export_zip_btn.click(
            fn=create_zip_export,
            inputs=[year, month, day, min_proba, min_stations_adv],
            outputs=export_file,
        )

        # Legacy gallery view
        legacy_btn.click(
            fn=load_images_and_table,
            inputs=[year, month, day, sort_by, min_proba, min_stations_adv],
            outputs=[gallery, gr.Dataframe(visible=False)],
        )

        legacy_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=gallery,
        )

        return demo


if __name__ == "__main__":
    demo = create_demo()
    # Launch with allowed_paths to serve files
    demo.launch(allowed_paths=[BASE_PATH])
