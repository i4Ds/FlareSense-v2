import pandas as pd
import torch
import yaml
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import wandb
from ecallisto_dataset import EcallistoDatasetBinary, custom_resize, remove_background
from ecallisto_model import GrayScaleResNet
from pred_live import (
    predict_from_to,
    load_model,
    create_logits,
    prepare_ecallisto_datasets,
    prepare_dataloaders,
)
from huggingface_hub import hf_hub_download

REPO_ID = "i4ds/flaresense-v2"
MODEL_FILENAME = "model.ckpt"
CONFIG_PATH = "configs/best_v2.yml"


def main(config):
    # Setup WandB API and download the artifact
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and configuration
    model, config = load_model(checkpoint_path, config)

    print("model loaded")

    # Prepare datasets and dataloaders
    ds = load_dataset(
        config["data"]["pred_path"],
        split=config["data"]["pred_split"],
        cache_dir="/mnt/nas05/data01/vincenzo/hu",
    )

    # Create ecallisto dataset and dataloader
    edb = prepare_ecallisto_datasets(ds, config)
    print(edb)

    dataloader = prepare_dataloaders(edb, config["general"]["batch_size"])

    # Predict probabilities
    preds = create_logits(model, dataloader, device)

    # Save a dataframe with the predictions
    ds = ds.select_columns(["datetime", "antenna"])
    df = ds.to_pandas()
    df["pred"] = preds
    df.to_csv(f"{config['data']['pred_path'].split('/')[1]}_test.csv")


if __name__ == "__main__":
    main("configs/best_v2.yml")
