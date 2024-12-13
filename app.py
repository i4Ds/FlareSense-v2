import gradio as gr
import glob, os
from datetime import datetime
import os, glob
import tempfile
import pandas as pd

BASE_PATH = os.path.join(os.getcwd(), "burst_plots")



def load_images(year, month, day, sort_by, min_proba):
    # Collect all images in all antenna subfolders for the given day
    search_path = os.path.join(BASE_PATH, year, month, day, "*", "*")
    files = glob.glob(search_path)
    if not files:
        return ["style/DALLE_ERROR.png"]

    # Parse filenames: "{proba}_{antenna}_{dd-mm-YYYY HH_MM_SS}.png"
    # Example: "90.47_AUSTRIA-UNIGRAZ_01_12-12-2024 07_29_13.png"
    img_data = []
    for f in files:
        base = os.path.basename(f)
        # Split by underscore: [proba, antenna_part(s), day-month-year, hour_min_sec.png]
        # There's an unknown number of underscores in antenna name, so we split and rejoin carefully:
        parts = base.split("_")
        proba_str = parts[0]  # "90.47"
        proba = float(proba_str)
        # Antenna might contain underscores, join all but last two parts back for antenna
        # Last two parts form date and time: "12-12-2024" and "07_29_13.png"
        antenna = "_".join(parts[1:-2])
        date_str = parts[-2]
        time_str = parts[-1].replace(".png", "")
        # Combine date_str and time_str into datetime
        dt_str = date_str + " " + time_str
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")
        img_data.append((f, proba, dt))

    if sort_by == "Probability":
        img_data.sort(key=lambda x: x[1], reverse=True)  # Sort by proba
    else:
        img_data.sort(key=lambda x: x[2], reverse=True)  # Sort by time

    # Min proba
    img_data = [x for x in img_data if x[1] >= min_proba]
    if len(img_data) == 0:
        return ["style/DALLE_ERROR.png"]

    return [(x[0], f"Probability: {x[1]:.2f}") for x in img_data]


def table_to_df(table_data):
    df = pd.DataFrame(table_data, columns=["Antenna", "Date", "Probability"])
    return df

def download_csv(year, month, day, sort_by, min_proba):
    # Return CSV for immediate download
    _, table_data = load_images_and_table(year, month, day, sort_by, min_proba)
    df = table_to_df(table_data)
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, f"FlareSense_BurstPlots_{year}_{month}_{day}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def load_images_and_table(year, month, day, sort_by, min_proba):
    img_data = load_images(year, month, day, sort_by, min_proba)  # from your code
    # Reload or reuse logic to produce table_data
    # Example: for each image path, parse antenna, dt, proba
    # Return both gallery data and table
    table_data = []
    search_path = os.path.join(BASE_PATH, year, month, day, "*", "*")
    for f in glob.glob(search_path):
        base = os.path.basename(f)
        parts = base.split("_")
        proba = float(parts[0])
        if proba < min_proba: 
            continue
        antenna = "_".join(parts[1:-2])
        dt_str = parts[-2] + " " + parts[-1].replace(".png","")
        dt = datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S")
        table_data.append([antenna, dt.strftime("%d-%m-%Y %H:%M:%S"), proba])

    return img_data, table_data

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
            <h1 style="margin-top:0;">FlareSense by i4ds</h1>
            <p style="font-size:1.1em;">
            A tool for predicting solar radio bursts.\n
            Select a date, sorting mode, and probability threshold. Click on a image to increase its size.\n
            Predictions update every 2 hours, using data from:\n
            <b>{", ".join(INSTRUMENT_LIST)}</b>.
            </p>
            <p style="font-size:0.9em;">
            For more info, refer to our <a href="https://placeholder.link.to.paper" target="_blank">paper</a>.
            For questions or comments, contact <a href="mailto:placeholder@placeholder.com" target="_blank">placeholder@placeholder.com</a>.
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
            object_fit="contain", elem_id="gallery", columns=[3], rows=[1]
        )
        table = gr.Dataframe(headers=["Antenna","Datetime","Probability"], wrap=True)
        load_btn.click(load_images_and_table, [year, month, day, sort_by, min_proba], [gallery, table])
        download_btn = gr.Button("Download CSV")
        download_file = gr.File()  # outputs a file to download

        download_btn.click(fn=download_csv,
                        inputs=[year, month, day, sort_by, min_proba],
                        outputs=download_file)

    demo.launch()
