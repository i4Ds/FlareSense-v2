import wandb
import torch
import yaml
from ecallisto_model import ResNet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from datasets import DatasetDict, load_dataset
import pandas as pd
from ecallisto_dataset import (
    EcallistoDatasetBinary,
    custom_resize,
    preprocess_spectrogram,
)


def create_probs(model, dataloader, device):
    model.eval()  # Ensure the model is in evaluation mode
    model.to(device)  # Send the model to the appropriate device
    class_1_prob = []

    with torch.no_grad():
        for inputs, _, _, _ in tqdm(dataloader):
            y_hat = model(inputs.to(device))
            probs, _ = model.calculate_prediction(y_hat)
            class_1_prob.extend(probs[:, 1].detach().cpu().tolist())

    return class_1_prob


def load_model(checkpoint_path, config_path, device):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Initialize the model
    model = ResNet(
        2,
        resnet_type=config["model"]["model_type"],
        optimizer_name="adam",
        learning_rate=1000,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model, config


def prepare_datasets(config):
    ds_train = load_dataset(config["data"]["train_path"], split="train")
    ds_valid = load_dataset(config["data"]["train_path"], split="validation")
    ds_test = load_dataset(config["data"]["train_path"], split="test")

    dd = DatasetDict()
    dd["train"] = ds_train
    dd["test"] = ds_test
    dd["validation"] = ds_valid
    return dd


def prepare_ecallisto_datasets(dd, config):
    resize_func = Compose(
        [lambda x: custom_resize(x, tuple(config["model"]["input_size"]))]
    )

    ds_train = EcallistoDatasetBinary(
        dd["train"],
        resize_func=resize_func,
        normalization_transform=preprocess_spectrogram,
    )
    ds_valid = EcallistoDatasetBinary(
        dd["validation"],
        resize_func=resize_func,
        normalization_transform=preprocess_spectrogram,
    )
    ds_test = EcallistoDatasetBinary(
        dd["test"],
        resize_func=resize_func,
        normalization_transform=preprocess_spectrogram,
    )

    return ds_train, ds_valid, ds_test


def prepare_dataloaders(ds_train, ds_valid, ds_test, batch_size):
    train_dataloader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=False, persistent_workers=False
    )
    val_dataloader = DataLoader(
        ds_valid, batch_size=batch_size, shuffle=False, persistent_workers=False
    )
    test_dataloader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, persistent_workers=False
    )
    return train_dataloader, val_dataloader, test_dataloader


def main(checkpoint_reference, config):
    # Setup WandB API and download the artifact
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    _ = artifact.download()

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and configuration
    model, config = load_model(artifact.file(), config, device)

    # Prepare datasets and dataloaders
    dd = prepare_datasets(config)

    # Convert datetime columns to pd.datetime
    df_train = pd.DataFrame(dd["train"])
    df_val = pd.DataFrame(dd["validation"])
    df_test = pd.DataFrame(dd["test"])

    for df in [df_train, df_val, df_test]:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d_%H-%M-%S")

    # Create ecallisto dataset and dataloader
    ds_train, ds_valid, ds_test = prepare_ecallisto_datasets(dd, config)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        ds_train, ds_valid, ds_test, config["general"]["batch_size"]
    )

    # Predict probabilities
    df_val["pred"] = create_probs(model, val_dataloader, device)
    df_test["pred"] = create_probs(model, test_dataloader, device)
    df_train["pred"] = create_probs(model, train_dataloader, device)

    # Save to CSV
    df_val.to_csv(f"{artifact.digest}_val.csv")
    df_test.to_csv(f"{artifact.digest}_test.csv")
    df_train.to_csv(f"{artifact.digest}_train.csv")


if __name__ == "__main__":
    main("vincenzo-timmel/FlareSense-v2/best_model:v130", "configs/t1000.yml")
