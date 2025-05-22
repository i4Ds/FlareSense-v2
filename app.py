import gradio as gr
import glob, os
from datetime import datetime, timedelta
import os, glob
import tempfile
import pandas as pd
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go


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
    return df


def download_csv(year, month, day, sort_by, min_proba, k) -> str:
    # Return CSV for immediate download
    _, df = load_images_and_table(year, month, day, sort_by, min_proba, k)
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, f"FlareSense_BurstPlots_{year}_{month}_{day}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_bursts(min_proba=0.5, days=30):
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

    # Load images
    img_data = load_images(table_data, sort_by)

    return img_data, filtered_table.round(2)


def get_current_date():
    current_datetime = datetime.now()
    current_year = str(current_datetime.year)
    current_month = f"{current_datetime.month:02d}"
    current_day = f"{current_datetime.day:02d}"
    return current_year, current_month, current_day


def create_demo():
    years = ["2025", "2024"]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    current_year, current_month, current_day = get_current_date()

    # Create the Gradio interface
    with gr.Blocks(title="FlareSense Burst Detection") as demo:
        gr.Markdown(
            f"""
            <div style="border:1px solid #ccc; padding:15px; border-radius:5px;">
            <h1 style="margin-top:0;">FlareSense by <a href="https://i4ds.ch/" target="_blank">i4ds@fhnw</a></h1>
            <p style="font-size:1.1em;">

            <b>A tool for detecting solar radio bursts on <a href="https://www.e-callisto.org/" target="_blank">E-callisto</a> Data.<br></b>
            Select a date, sorting mode and minimum number of stations. <br>

            Predictions update every 2 hours, using data from:<br>
            <b>{", ".join(INSTRUMENT_LIST)}</b>.
            </p>
            <p style="font-size:0.9em;">
            For more information about the stations, refer to <a href="https://www.e-callisto.org/stations.html" target="_blank">E-Callisto Stations</a>.<br>
            For questions or comments, contact <a href="mailto:vincenzo.timmel@fhnw.ch" target="_blank">vincenzo.timmel@fhnw.ch</a>.        
            For more information about the project, refer to our <a href="https://placeholder.link.to.paper" target="_blank">paper</a>.<br>
            </p>
            </div>
            """
        )

        with gr.Row():
            year = gr.Dropdown(choices=years, label="Year", value=current_year)
            month = gr.Dropdown(choices=months, label="Month", value=current_month)
            day = gr.Dropdown(choices=days, label="Day", value=current_day)

            # technically, can also be from a dropdown menu.
            sort_by = gr.State(SORT_BY_COLUMN)
            min_proba = gr.State(MIN_PROBABILITY)

            k_stations = gr.State(1)  # default value
        load_btn = gr.Button("Load Images")

        gallery = gr.Gallery(
            object_fit="fill", elem_id="gallery", columns=[3], rows=[1], height="auto"
        )
        table = gr.Dataframe(
            headers=["Datetime", "Instrument Location", "Confidence"], wrap=False
        )
        # Load images of today once
        demo.load(
            fn=load_images_and_table,  # same function
            inputs=[
                year,
                month,
                day,  # date dropdowns
                sort_by,
                min_proba,
                k_stations,
            ],
            outputs=[gallery, table],
        )

        # Load images on button click
        load_btn.click(
            load_images_and_table,
            [year, month, day, sort_by, min_proba, k_stations],
            [gallery, table],
        )

        # Load once
        download_btn = gr.Button("Download CSV")
        download_file = gr.File()  # outputs a file to download

        download_btn.click(
            fn=download_csv,
            inputs=[year, month, day, sort_by, min_proba, k_stations],
            outputs=download_file,
        )

        # Bar Plot with number of bursts per day
        line_plot = gr.Plot()
        demo.load(
            fn=plot_bursts,
            inputs=None,
            outputs=line_plot,
        )

        return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(allowed_paths=[BASE_PATH])
