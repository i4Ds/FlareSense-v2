import wandb
import torch
import yaml
from ecallisto_model import (
    ResNet18,
    create_normalize_function,
    create_unnormalize_function,
)

import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from ecallisto_dataset import (
    EcallistoDatasetBinary,
)
from datasets import DatasetDict, load_dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# Download Model
checkpoint_reference = "vincenzo-timmel/FlareSense-v2/best_model:v6"

api = wandb.Api()
artifact = api.artifact(checkpoint_reference)
artifact_dir = artifact.download()

checkpoint = torch.load(
    "artifacts/best_model:v6/epoch=9-step=7494.ckpt", map_location="cpu"
)

# load checkpoint and model and helper functions
with open("antenna_stats.yaml", "r") as file:
    antenna_stats = yaml.safe_load(file)
normalize_transform = create_normalize_function(
    antenna_stats=antenna_stats, simple=False
)

unnormalize_img = create_unnormalize_function(antenna_stats)
model = ResNet18(2, unnormalize_img=unnormalize_img)
model.load_state_dict(checkpoint["state_dict"])
device = "cuda" if torch.cuda.is_available() else "cpu"


ds = load_dataset("i4ds/radio-sunburst-ecallisto-binary-cleaned-4")

dd = DatasetDict()
dd["train"] = ds["train"]
dd["test"] = ds["test"]
dd["validation"] = ds["validation"]

# Define normalization
base_transform = Compose(
    [
        Resize([224, 224]),  # Resize the image
    ]
)


## Evaluate
ds_test = EcallistoDatasetBinary(
    dd["test"],
    base_transform=base_transform,
    normalization_transform=normalize_transform,
)
test_dataloader = DataLoader(
    ds_test,
    batch_size=8,
    num_workers=4,
    shuffle=True,  # To randomly log images
    persistent_workers=False,
)

# Argument parser for config file path
parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration yaml file."
)
args = parser.parse_args()

# Load the configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)
# Setup wandb
wandb.init(entity="vincenzo-timmel", config=config)
wandb_logger = WandbLogger(log_model=False)

# Setup trainer
trainer = Trainer(
    accelerator=device,
    max_epochs=config["general"]["max_epochs"],
    logger=wandb_logger,
    enable_progress_bar=False,
    val_check_interval=200,
)
trainer.test(model, test_dataloader)
