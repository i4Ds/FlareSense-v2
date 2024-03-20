# Modeling
import argparse
import sys

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose, Resize

import wandb
from ecallisto_dataset import (
    EcallistoDataset,
    randomly_reduce_class_samples,
    EcallistoDatasetBinary,
)
from ecallisto_model import (
    EfficientNet,
    create_normalize_function,
    create_unnormalize_function,
)


if __name__ == "__main__":
    print(f"PyTorch version {torch.__version__}")
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("GPU is not available.")
        device = "cpu"

    # Mixed precision
    torch.set_float32_matmul_precision("high")

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
    wandb_logger = WandbLogger(log_model="all")

    # Overwrite config with wandb config (for sweep etc.)
    del config
    config = wandb.config

    # Create dataset
    ds = load_dataset("i4ds/radio-sunburst-ecallisto")

    dd = DatasetDict()
    dd["train"] = randomly_reduce_class_samples(
        ds["train"],
        config["data"]["train_class_to_reduce"],
        config["data"]["reduction_fraction"],
    )
    dd["test"] = ds["test"]
    dd["validation"] = ds["validation"]

    size = tuple(config["model"]["input_size"])

    # Transforms
    base_transform = Compose(
        [
            Resize(size),  # Resize the image
        ]
    )
    if config["data"]["use_augmentation"]:
        data_augm_transform = Compose(
            [
                FrequencyMasking(freq_mask_param=config["data"]["freq_mask_param"]),
                TimeMasking(time_mask_param=config["data"]["time_mask_param"]),
            ]
        )
    else:
        data_augm_transform = None

    # Define normalization
    with open("antenna_stats.yaml", "r") as file:
        antenna_stats = yaml.safe_load(file)
    normalize_transform = create_normalize_function(antenna_stats=antenna_stats)

    # Data Loader
    ds_train = EcallistoDatasetBinary(
        dd["train"],
        base_transform=base_transform,
        data_augm_transform=data_augm_transform,
        normalization_transform=normalize_transform,
    )
    ds_valid = EcallistoDatasetBinary(
        dd["validation"],
        base_transform=base_transform,
        normalization_transform=normalize_transform,
    )

    # Create Data loader
    sample_weights = (
        ds_train.get_sample_weights()
    )  # Never binary, because we also want to detect rare radio sunbursts.
    if config["general"]["use_random_sampler"]:
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
    else:
        sampler = None

    train_dataloader = DataLoader(
        ds_train,
        sampler=sampler,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=False if sampler is not None else True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        ds_valid,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
    )

    # Checkpoint to save the best model based on the lowest validation loss
    checkpoint_callback_rafp = ModelCheckpoint(
        monitor="val_rafp",
        dirpath=wandb_logger.experiment.dir,
        filename="efficientnet-rafp-{epoch:02d}-{step:05d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Setup Model
    cw = torch.tensor(ds_train.get_class_weights(), dtype=torch.float)
    unnormalize_img = create_unnormalize_function(antenna_stats)
    model = EfficientNet(
        n_classes=2,  # Binary
        class_weights=(cw if config["general"]["use_class_weights"] else None),
        dropout=config["model"]["dropout"],
        learnig_rate=config["model"]["lr"],
        min_precision=config["general"]["min_precision"],
        unnormalize_img=unnormalize_img,
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["general"]["max_epochs"],
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback_rafp,
            EarlyStopping(monitor="val_loss", mode="min", patience=20),
        ],
        val_check_interval=200,
    )

    # Train
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    ## Evaluate
    ds_test = EcallistoDatasetBinary(
        dd["test"],
        base_transform=base_transform,
        normalization_transform=normalize_transform,
    )
    test_dataloader = DataLoader(
        ds_test,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=True,  # To randomly log images
        persistent_workers=False,
    )
    trainer.test(model, test_dataloader, ckpt_path="best")
