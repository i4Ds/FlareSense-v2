import torch
import yaml
import os
import glob
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
import numpy as np
import shutil
import plotly.graph_objects as go

torch.set_float32_matmul_precision("high")


def create_logits(model: GrayScaleResNet, dataloader: DataLoader, device: str):
    """Generate logits for all samples in a DataLoader."""
    model.eval()
    model.to(device)
    binary_logits = []
    with torch.no_grad():
        for inputs, _, _, _ in tqdm(dataloader):
            y_hat = model(inputs.to(device)).squeeze(dim=1)
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
        learning_rate=1000,
        label_smoothing=0.0,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config


def prepare_ecallisto_datasets(ds, config: dict):
    """Prepare EcallistoDatasetBinary with resizing and normalization."""
    resize_func = Compose(
        [lambda x: custom_resize(x, tuple(config["model"]["input_size"]))]
    )
    ds = ds.add_column("dummy_label", [0] * len(ds))
    edb = EcallistoDatasetBinary(
        ds,
        label_name="dummy_label",
        resize_func=resize_func,
        normalization_transform=remove_background,
    )
    return edb


def prepare_dataloaders(ds: EcallistoDatasetBinary, batch_size: int):
    """Create DataLoader from dataset."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, persistent_workers=False
    )


def cleanup_parquets(directory: str):
    """Delete parquet files in a directory."""
    files = glob.glob("*.parquet", root_dir=directory, recursive=True)
    for f in files:
        os.remove(f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    repo_id = "i4ds/flaresense"
    filename = "model.ckpt"
    config_path = "configs/relabeled_data_best.yml"
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    create_overlapping_parquets(
        now - timedelta(hours=12),
        now,
        [
            "Australia-ASSA_02",
            "Australia-ASSA_62",
            "ALASKA-HAARP_62",
            "AUSTRIA-UNIGRAZ_01",
            "BIR_01",
            "GERMANY-DLR_63",
            "GERMANY-ESSEN_58",
            "HUMAIN_59",
            "ITALY-Strassolt_01",
            "NORWAY-EGERSUND_01",
            "TRIEST_57"
        ],
        "flaresense-data",
    )

    ds = load_radio_dataset("flaresense-data/")
    model, config = load_model(checkpoint_path, config_path)
    ds_e = prepare_ecallisto_datasets(ds, config)
    data_loader = prepare_dataloaders(ds_e, 32)
    preds = create_logits(model, data_loader, "cuda")
    ds = ds.add_column("pred", preds)
    ds_bursts = ds.filter(lambda x: x["pred"] > 0).select_columns(
        ["datetime", "antenna", "pred", "path"]
    )
    df_bursts = ds_bursts.to_pandas()
    df_bursts["proba"] = df_bursts["pred"].apply(lambda x: sigmoid(x))

    # Create plots
    for i, row in df_bursts.iterrows():
        data = get_ecallisto_data(
            row["datetime"], row["datetime"] + timedelta(minutes=15), row["antenna"]
        )[row["antenna"]]
        fig = plot_spectrogram(subtract_constant_background(data).clip(0, 16))

        path = f'burst_plots/{row["antenna"]}/{row["proba"]*100:.2f}_{row["antenna"]}_{row["datetime"].strftime("%d-%m-%Y_%H-%M-%S")}.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_image(path)

    # Cleanup
    shutil.rmtree("flaresense-data")
