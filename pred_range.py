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

from ecallisto_dataset import EcallistoDatasetBinary, custom_resize, remove_background
from ecallisto_model import GrayScaleResNet
from ecallisto_ng.plotting.plotting import plot_spectrogram
from ecallisto_ng.data_processing.utils import subtract_constant_background
from ecallisto_ng.data_download.downloader import get_ecallisto_data
from pred_live import (
    create_logits,
    load_model,
    prepare_ecallisto_datasets,
    prepare_dataloaders,
    predict_from_to,
)
import tempfile
import shutil
import pandas as pd

# Model Parameters
REPO_ID = "i4ds/flaresense"
MODEL_FILENAME = "model.ckpt"
CONFIG_PATH = "configs/relabeled_data_best.yml"
T = 1.006  # Temperature parameter
torch.set_float32_matmul_precision("high")

# Parameters
INSTRUMENT_LIST = [
    "ALMATY_59",
    "Australia-ASSA_57",
    "Australia-ASSA_63",
    "AUSTRIA-UNIGRAZ_01",
    "EGYPT-SpaceAgency_01",
    "GERMANY-DLR_63",
    "HUMAIN_59",
    "INDIA-GAURI_59",
    "INDIA-UDAIPUR_03",
    "MEXART_59",
    "MEXICO-LANCE-B_62",
    "NORWAY-EGERSUND_01",
    "ROMANIA_01",
    "ROSWELL-NM_59",
    "SSRT_59",
    "TRIEST_57",
    "ALASKA-HAARP_62",
    "KASI_59",
    "Malaysia-Banting_01",
    "MONGOLIA-UB_01",
    "UZBEKISTAN_01",
    "INDONESIA_59",
    "ITALY-Strassolt_01",
    "MONGOLIA-UB_01",
    "SWISS-IRSOL_01",
    "SWISS-Landschlacht_62",
]

if __name__ == "__main__":
    BASE_PATH = "output_burst_images"

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

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

    # Predict between two ranges
    start_datetime = datetime(2024, 12, 18, 0, 0, 0, tzinfo=timezone.utc)
    end_datetime = datetime(2024, 12, 18, 22, 0, 0, tzinfo=timezone.utc)

    # Split it up into two-hour steps
    for start_time in pd.date_range(
        start_datetime, end_datetime, freq="2h", inclusive="both"
    ):
        end_time = start_time + timedelta(hours=2)
        predict_from_to(start_time, end_time, model, config)
