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


def create_logits(model: GrayScaleResNet, dataloader, device):
    model.eval()  # Ensure the model is in evaluation mode
    model.to(device)  # Send the model to the appropriate device
    binary_logits = []

    print("Starting prediction")
    with torch.no_grad():
        for inputs, _, _, _ in tqdm(dataloader):
            y_hat = model(inputs.to(device)).squeeze(dim=1)
            binary_logits.extend(y_hat.cpu().tolist())

    return binary_logits


def load_model(checkpoint_path, config_path):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Initialize the model
    model = GrayScaleResNet(
        1,
        resnet_type=config["model"]["model_type"],
        optimizer_name="adam",
        learning_rate=1000,
        label_smoothing=0.0,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    return model, config


def prepare_ecallisto_datasets(ds, config):
    resize_func = Compose(
        [lambda x: custom_resize(x, tuple(config["model"]["input_size"]))]
    )
    print("Wohoho")
    ds = ds.add_column("dummy_label", [0] * len(ds))
    edb = EcallistoDatasetBinary(
        ds,
        label_name="dummy_label",
        resize_func=resize_func,
        normalization_transform=remove_background,
    )
    return edb


def prepare_dataloaders(ds, batch_size):
    dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, persistent_workers=False
    )
    return dataloader


def main(checkpoint_reference, config):
    # Setup WandB API and download the artifact
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    _ = artifact.download()

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and configuration
    print(artifact.file())
    model, config = load_model(artifact.file(), config)

    print("model loaded")

    # Prepare datasets and dataloaders
    ds = load_dataset(config["data"]["pred_path"], split=config["data"]["pred_split"])

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
    main("vincenzo-timmel/FlareSense-v2/final_model:v7", "configs/best_v2.yml")
