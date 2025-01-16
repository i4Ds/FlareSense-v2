import gradio as gr
import glob, os
from datetime import datetime, timedelta
import os, glob
import tempfile
import pandas as pd
from typing import List, Tuple
import plotly.express as px


BASE_PATH = os.path.join("/mnt/nas05/data01/vincenzo/ecallisto/burst_live_images")

from pred_live import INSTRUMENT_LIST


def load_images(table, sort_by) -> List:
    if sort_by == "Probability":
        table = table.sort_values(by="Probability", ascending=False)
    else:
        table = table.sort_values(by="Datetime", ascending=False)

    img_data = []
    for _, row in table.iterrows():
        img_data.append((row["Path"], row["Probability"]))

    if len(img_data) == 0:
        return ["style/DALLE_ERROR.png"]

    return [(x[0], f"Probability: {x[1]:.2f}") for x in img_data]


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
        table_data, columns=["Datetime", "Instrument Location", "Probability", "Path"]
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


def plot_bursts(min_proba, k, days=30):
    dfs = []
    for i in range(days):
        d = datetime.now() - timedelta(days=i)
        y, m, da = str(d.year), f"{d.month:02d}", f"{d.day:02d}"
        _, df = load_images_and_table(y, m, da, "Datetime", min_proba, k)
        df = load_image_paths(y, m, da, min_proba)
        dfs.append(df)

    if not dfs:
        return {}

    all_data = pd.concat(dfs, ignore_index=True)
    # Group by day instead of hour
    all_data["Day"] = all_data["Datetime"].dt.floor("D")
    daily_counts = all_data.groupby("Day").size().reset_index(name="Count")

    # Create a bar plot
    fig = px.bar(
        daily_counts,
        x="Day",
        y="Count",
        title="Number of Bursts per Day (Last 14 Days)",
        labels={"Day": "Date", "Count": "Bursts"},
    )

    # Make it look nicer
    fig.update_traces(marker_color="royalblue")
    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title="Date",
        yaxis_title="Bursts Detected",
        font=dict(size=14),
    )
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
                "Probability": "mean",
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

    with gr.Blocks(title="FlareSense Burst Detection") as demo:
        gr.Markdown(
            f"""
            <div style="border:1px solid #ccc; padding:15px; border-radius:5px;">
            <h1 style="margin-top:0;">FlareSense by <a href="https://i4ds.ch/" target="_blank">i4ds@fhnw</a></h1>
            <p style="font-size:1.1em;">

            
            <b>A tool for detecting solar radio bursts on <a href="https://www.e-callisto.org/" target="_blank">E-callisto</a> Data.<br></b>
            Select a date, sorting mode, probability threshold, and minimum number of stations. Click on an image to increase its size. The Barplot is unaffected by the minimum Probability and minimum number of stations.<br>

            
            Predictions update every 2 hours, using data from:<br>
            <b>{", ".join(INSTRUMENT_LIST)}</b>.
            </p>
            <p style="font-size:0.9em;">
            For more info, refer to our <a href="https://placeholder.link.to.paper" target="_blank">paper</a>.<br>
            For questions or comments, contact <a href="mailto:vincenzo.timmel@fhnw.ch" target="_blank">vincenzo.timmel@fhnw.ch</a>.
            </p>
            </div>
            """
        )

        with gr.Row():
            year = gr.Dropdown(choices=years, label="Year")
            month = gr.Dropdown(choices=months, label="Month")
            day = gr.Dropdown(choices=days, label="Day")

            demo.load(get_current_date, inputs=None, outputs=[year, month, day])

            sort_by = gr.Dropdown(
                choices=["Probability", "Time"], value="Probability", label="Sort By"
            )
            min_proba = gr.Slider(
                minimum=50.0,
                maximum=100.0,
                value=50.0,
                label="Minimum Probability",
                info="Filter by minimum probability",
            )
            k_stations = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Minimum Number of Stations (k)",
                info="At least k stations must have detected the burst",
            )

        load_btn = gr.Button("Load Images")
        # Show the line plot below the load images parameters when the app is opened
        line_plot = gr.Plot()
        demo.load(
            fn=plot_bursts,
            inputs=[min_proba, k_stations],
            outputs=line_plot,
        )
        gallery = gr.Gallery(
            object_fit="fill", elem_id="gallery", columns=[3], rows=[1], height="auto"
        )
        table = gr.Dataframe(
            headers=["Datetime", "Instrument Location", "Probability"], wrap=False
        )
        load_btn.click(
            load_images_and_table,
            [year, month, day, sort_by, min_proba, k_stations],
            [gallery, table],
        )
        download_btn = gr.Button("Download CSV")
        download_file = gr.File()  # outputs a file to download

        download_btn.click(
            fn=download_csv,
            inputs=[year, month, day, sort_by, min_proba, k_stations],
            outputs=download_file,
        )

        return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(allowed_paths=[BASE_PATH])
