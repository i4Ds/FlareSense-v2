import torch
import yaml
import os
from datetime import datetime, timedelta, timezone
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from ecallisto_ng.data_download.hu_dataset import (
    create_overlapping_parquets,
    load_radio_dataset,
)

from ecallisto_dataset import (
    EcallistoDatasetBinary,
    custom_resize,
    remove_background,
    normal_resize,
)
from ecallisto_model import GrayScaleResNet
from ecallisto_ng.plotting.plotting import plot_spectrogram
from ecallisto_ng.data_processing.utils import subtract_constant_background
from ecallisto_ng.data_download.downloader import get_ecallisto_data
import numpy as np
import tempfile
import shutil
import time

# Model Parameters
REPO_ID = "i4ds/flaresense-v2"
MODEL_FILENAME = "model.ckpt"
CONFIG_PATH = "configs/best_v2.yml"
T = 0.4974  # Temperature parameter
torch.set_float32_matmul_precision("high")


# Parameters
INSTRUMENT_LIST = [
    "ALASKA-ANCHORAGE_91",
    "ALASKA-COHOE_63",
    "ALASKA-HAARP_62",
    "ALGERIA-CRAAG_59",
    "ALMATY_59",
    "AUSTRIA-UNIGRAZ_01",
    "Australia-ASSA_57",
    "Australia-ASSA_63",
    "BIR_01",
    "EGYPT-Alexandria_02",
    "EGYPT-SpaceAgency_01",
    "GERMANY-DLR_63",
    "GLASGOW_01",
    "GREENLAND_62",
    "HUMAIN_59",
    "HURBANOVO_59",
    "INDIA-GAURI_59",
    "INDIA-OOTY_02",
    "INDIA-UDAIPUR_03",
    "INDONESIA_59",
    "ITALY-Strassolt_01",
    "KASI_59",
    "MEXART_59",
    "MEXICO-FCFM-UNACH_62",
    "MEXICO-LANCE-B_62",
    "MONGOLIA-UB_01",
    "MRO_59",
    "MRO_61",
    "Malaysia-Banting_01",
    "NORWAY-EGERSUND_01",
    "ROMANIA_01",
    "ROSWELL-NM_59",
    "SRI-Lanka_59",
    "SSRT_59",
    "SWISS-IRSOL_01",
    "SWISS-Landschlacht_62",
    "TRIEST_57",
    "UZBEKISTAN_01",
]


def sigmoid(x, T=T):
    return 1 / (1 + np.exp(-x / T))


def create_logits(model: GrayScaleResNet, dataloader: DataLoader):
    """
    Generate logits for all samples in a DataLoader.

    Args:
        model (GrayScaleResNet): The model used to generate logits.
        dataloader (DataLoader): DataLoader providing batches of input data.

    Returns:
        list: A list of logits for all samples in the DataLoader.
    """
    model.eval()
    binary_logits = []
    with torch.no_grad():
        for inputs, _, _, _ in tqdm(dataloader):
            y_hat = model(inputs.to(model.device)).squeeze(dim=1)
            binary_logits.extend(y_hat.cpu().tolist())
    return binary_logits


def load_model(checkpoint_path: str, config_path: str):
    """Load model and config."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model = GrayScaleResNet(
        n_classes=1,
        resnet_type=config["model"]["model_type"],
        optimizer_name="adam",
        learning_rate=1000,  # Dummy value
        label_smoothing=0.0,  # Dummy value
        max_epochs=1000,  # Dummy value
        warmup_epochs=10,  # Dummy value
    )
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model, config


def prepare_ecallisto_datasets(ds, config: dict, live_mode: bool = False):
    """Prepare EcallistoDatasetBinary with resizing and normalization."""
    if config["data"]["custom_resize"]:
        resize_func = Compose(
            [
                lambda x: custom_resize(x, tuple(config["model"]["input_size"])),
            ]
        )
    else:
        resize_func = Compose(
            [
                lambda x: normal_resize(x, tuple(config["model"]["input_size"])),
            ]
        )
    edb = EcallistoDatasetBinary(
        ds,
        label_name="dummy_label" if live_mode else config["data"]["test_label_name"],
        resize_func=resize_func,
        normalization_transform=remove_background,
    )
    return edb


def prepare_dataloaders(ds: EcallistoDatasetBinary, batch_size: int):
    """Create DataLoader from dataset."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, persistent_workers=False
    )


def predict_from_to(start_datetime, end_datetime, model, config, base_path=None):
    if base_path is None:
        base_path = BASE_PATH
    # Temporary directory for parquet data
    tmp_dir = tempfile.mkdtemp()

    try:
        # Create parquet data from instruments
        create_overlapping_parquets(
            start_datetime,
            end_datetime,
            INSTRUMENT_LIST,
            tmp_dir,
            download_from_local=True,
        )

        # Load dataset and model
        ds = load_radio_dataset(tmp_dir)
        if ds is None:
            raise ValueError("No data found in parquet files.")

        # Prepare dataset and dataloaders
        ds_e = prepare_ecallisto_datasets(ds, config, live_mode=True)
        print(f"Predicting on {len(ds_e)} samples.")

        data_loader = prepare_dataloaders(ds_e, batch_size=8)

        # Generate predictions
        preds = create_logits(model, data_loader)
        ds = ds.add_column("pred", preds)

        # Filter bursts
        ds_bursts = ds.filter(lambda x: x["pred"] > 0).select_columns(
            ["datetime", "antenna", "pred"]
        )

        # To dataframe for easier processing
        df_bursts = ds_bursts.to_pandas()

        print(f"Number of bursts: {len(df_bursts)}")

        # Create probabilities from logits
        df_bursts["proba"] = df_bursts["pred"].apply(lambda x: sigmoid(x))

        # Generate and save plots
        for _, row in df_bursts.iterrows():
            # Create figure
            data = get_ecallisto_data(
                row["datetime"], row["datetime"] + timedelta(minutes=15), row["antenna"]
            )[row["antenna"]]
            fig = plot_spectrogram(subtract_constant_background(data).clip(0, 16))

            # Save figure
            year = row["datetime"].year
            month = f'{row["datetime"].month:02d}'
            day = f'{row["datetime"].day:02d}'
            out_dir = f'{base_path}/{year}/{month}/{day}/{row["antenna"]}'
            out_name = f'{row["proba"]*100:.2f}_{row["antenna"]}_{row["datetime"].strftime("%d-%m-%Y_%H-%M-%S")}.png'
            out_path = os.path.join(out_dir, out_name)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.write_image(out_path)

    finally:
        shutil.rmtree(tmp_dir)
        print(4 * "-" + " END " + 4 * "-")


if __name__ == "__main__":
    from app import BASE_PATH

    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {CONFIG_PATH}")
    print(f"Using model from {REPO_ID}")

    # Load model
    model, config = load_model(checkpoint_path, CONFIG_PATH)

    # Put model on device
    model.to(device)

    while True:
        print(4 * "=" + " PREDICTION " + 4 * "=")
        now = datetime.now(timezone.utc)

        # Calculate the next 30-minute mark
        next_minute = ((now.minute // 30) + 1) * 30
        if next_minute == 60:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
        else:
            next_run = now.replace(minute=next_minute, second=0, microsecond=0)

        # Sleep for the time until the next 30-minute mark
        ts = (next_run - now).total_seconds()
        print(
            f"Next run at: {next_run} UTC. I will sleep for {ts} seconds.", flush=True
        )
        time.sleep(ts)

        print(4 * "-" + " START " + 4 * "-")
        # Prepare time range: predict on the last 1 hour
        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_time = end_time - timedelta(hours=1)

        # Print some logs, like start time and end time
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")

        # Predict
        predict_from_to(start_time, end_time, model, config)
