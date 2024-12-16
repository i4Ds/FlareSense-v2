import gradio as gr
import glob, os
from datetime import datetime
import os, glob
import tempfile
import pandas as pd

BASE_PATH = os.path.join("/mnt/nas05/data01/vincenzo/ecallisto/burst_live_images")


def load_images(table, sort_by):
    img_data = []
    for _, row in table.iterrows():
        img_data.append((row["Path"], row["Probability"]))

    # Min proba
    if len(img_data) == 0:
        return ["style/DALLE_ERROR.png"]

    # Sort
    if sort_by == "Probability":
        img_data.sort(key=lambda x: x[1], reverse=True)
    else:
        img_data.sort(key=lambda x: x[0], reverse=True)

    return [(x[0], f"Probability: {x[1]:.2f}") for x in img_data]


def load_image_paths(year, month, day, min_proba):
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
        table_data, columns=["Datetime", "Antenna", "Probability", "Path"]
    ).sort_values(by="Datetime", ascending=True)
    return df


def download_csv(year, month, day, min_proba):
    # Return CSV for immediate download
    df: pd.DataFrame = load_image_paths(year, month, day, min_proba).drop(
        columns=["Path"]
    )
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, f"FlareSense_BurstPlots_{year}_{month}_{day}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def load_images_and_table(year, month, day, sort_by, min_proba):
    table_data = load_image_paths(year, month, day, min_proba)
    img_data = load_images(table_data, sort_by)

    table_data = (
        table_data.drop(columns=["Path"])
        .sort_values(by=["Datetime", "Antenna"], ascending=[True, True])
        .groupby(["Datetime"])
        .agg(
            {
                "Datetime": "first",
                "Antenna": lambda x: ", ".join(set(x)),
                "Probability": "mean",
            }
        )
    )

    return img_data, table_data.round(2)


if __name__ == "__main__":
    # Glob folder structure
    from pred_live import INSTRUMENT_LIST

    years = ["2025", "2024"]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    current_date = datetime.now()
    current_year = str(current_date.year)
    current_month = str(current_date.month)
    current_day = str(current_date.day)

    with gr.Blocks(title="FlareSense Burst Detection") as demo:
        gr.Markdown(
            f"""
            <div style="border:1px solid #ccc; padding:15px; border-radius:5px;">
            <h1 style="margin-top:0;">FlareSense by <a href="https://i4ds.ch/" target="_blank">i4ds@fhnw</a></h1>
            <p style="font-size:1.1em;">
            A tool for detecting solar radio bursts on <a href="https://www.e-callisto.org/" target="_blank">E-callisto</a> Data.\n
            Select a date, sorting mode, and probability threshold. Click on a image to increase its size.\n
            Predictions update every 2 hours, using data from:\n
            <b>{", ".join(INSTRUMENT_LIST)}</b>.
            </p>
            <p style="font-size:0.9em;">
            For more info, refer to our <a href="https://placeholder.link.to.paper" target="_blank">paper</a>.
            For questions or comments, contact <a href="mailto:vincenzo.timmel@fhnw.ch" target="_blank">vincenzo.timmel@fhnw.ch</a>.
            </p>
            </div>
            """
        )

        with gr.Row():
            year = gr.Dropdown(choices=years, value=current_year, label="Year")
            month = gr.Dropdown(choices=months, value=current_month, label="Month")
            day = gr.Dropdown(choices=days, value=current_day, label="Day")
            sort_by = gr.Dropdown(
                choices=["Probability", "Time"], value="Probability", label="Sort By"
            )
            min_proba = gr.Slider(
                minimum=50.0,
                maximum=100.0,
                value=70.0,
                label="Minimum Probability",
                info="Filter by minimum probability",
            )

        load_btn = gr.Button("Load Images")
        gallery = gr.Gallery(
            object_fit="fill", elem_id="gallery", columns=[3], rows=[1], height="auto"
        )
        table = gr.Dataframe(headers=["Datetime", "Antenna", "Probability"], wrap=True)
        load_btn.click(
            load_images_and_table,
            [year, month, day, sort_by, min_proba],
            [gallery, table],
        )
        download_btn = gr.Button("Download CSV")
        download_file = gr.File()  # outputs a file to download

        download_btn.click(
            fn=download_csv,
            inputs=[year, month, day, min_proba],
            outputs=download_file,
        )

    demo.launch(allowed_paths=[BASE_PATH])
