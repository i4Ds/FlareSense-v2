import torch

import os
from datetime import datetime, timedelta, timezone

from huggingface_hub import hf_hub_download


import pandas as pd
from pred_live import (
    predict_from_to,
    load_model,
    REPO_ID,
    MODEL_FILENAME,
    CONFIG_PATH,
)
from app import BASE_PATH

# Model Parameters
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
    start_datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_datetime = datetime(2024, 12, 31, 22, 0, 0, tzinfo=timezone.utc)

    # Split it up into two-hour steps
    for start_time in pd.date_range(
        start_datetime, end_datetime, freq="2h", inclusive="both"
    ):
        end_time = start_time + timedelta(hours=2)
        predict_from_to(start_time, end_time, model, config, BASE_PATH)
