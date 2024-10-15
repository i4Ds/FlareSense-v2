import torch

# Modeling
# Visualization
import torch
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision import models

import wandb

# Modeling
import argparse
import os

import torch
import yaml
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
from ecallisto_dataset import (
    CustomSpecAugment,
    EcallistoDatasetBinary,
    TimeWarpAugmenter,
    custom_resize,
    remove_background,
)

RESNET_DICT = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet52": models.resnet50,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, lambda_bt=5e-3):
    """
    Computes the Barlow Twins loss between two sets of embeddings.

    Args:
        z1 (torch.Tensor): Embeddings from the first view.
        z2 (torch.Tensor): Embeddings from the second view.
        lambda_bt (float): Weighting factor for the off-diagonal loss.

    Returns:
        torch.Tensor: The Barlow Twins loss.
    """
    # Normalize the embeddings
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)
    N = z1.size(0)  # Batch size

    # Compute cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / N

    # Compute loss
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    loss = on_diag + lambda_bt * off_diag
    return loss


class ResNetBarlow(LightningModule):
    def __init__(
        self,
        resnet_type,
        batch_size,
        optimizer_name,
        learning_rate,
        augmentation_func,
        model_weights: str = None,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        lambda_bt: float = 5e-3,
    ):
        """
        Initializes the ResNetBarlow model for self-supervised learning using Barlow Twins.

        Args:
            resnet_type (str): Type of ResNet to use (e.g., 'resnet50').
            batch_size (int): Training batch size.
            optimizer_name (str): Optimizer to use ('adam' or 'adamw').
            learning_rate (float): Learning rate for the optimizer.
            augmentation_func (callable): Data augmentation function.
            model_weights (str): Pretrained weights to use.
            projection_dim (int): Output dimension of the projection head.
            projection_hidden_dim (int): Hidden layer dimension of the projection head.
            lambda_bt (float): Weighting factor for the Barlow Twins loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.augmentation_func = augmentation_func
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

        # Get model
        resnet_cls = RESNET_DICT.get(resnet_type)

        ### BARLOW STUFF
        # Initialize ResNet without the final classification layer
        self.encoder = resnet_cls(weights=model_weights)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection head as per Barlow Twins
        self.projector = nn.Sequential(
            nn.Linear(in_features, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )
        # Hyperparameters for Barlow Twins loss
        self.lambda_bt = lambda_bt

    def forward(self, x):
        """
        Forward pass through the encoder and projection head.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        # Ensure input has 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Encode and project
        features = self.encoder(x)
        projections = self.projector(features)
        return projections

    def validation_step(self, batch, batch_idx):
        """
        Validation step for Barlow Twins self-supervised learning.

        Args:
            batch (tuple): Batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed validation loss.
        """
        x, _, _, _ = batch  # Assuming batch = (x, _, _, _)

        # Create two augmented views
        xt1 = self.augmentation_func(x)
        xt2 = self.augmentation_func(x)

        # Obtain projections
        z1 = self(xt1)
        z2 = self(xt2)

        # Compute Barlow Twins loss
        loss = barlow_twins_loss(z1, z2, lambda_bt=self.lambda_bt)

        # Logging
        self.log(
            "val_barlow_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def training_step(self, batch, batch_idx):
        x, _, _, _ = batch

        # Create augmentations
        xt1 = self.augmentation_func(x)
        xt2 = self.augmentation_func(x)

        # Pass through the model to get embeddings
        z1 = self(xt1)
        z2 = self(xt2)

        # Barlow Twins loss
        barlow_loss = barlow_twins_loss(z1, z2)

        # Logging
        self.log("barlow_loss", barlow_loss, on_step=True, batch_size=self.batch_size)

        return barlow_loss

    def configure_optimizers(self):
        return OPTIMIZERS[self.optimizer_name](
            params=self.parameters(), lr=self.learning_rate
        )


if __name__ == "__main__":
    print(f"PyTorch version {torch.__version__}")
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_id = os.environ["SLURM_JOB_GPUS"]
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(
            f"GPU is available: {device_name} (Device ID: {device_id}). I have {os.cpu_count()} cores available."
        )
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
    wandb.init(
        entity="vincenzo-timmel",
        config=config,
        settings=wandb.Settings(
            _stats_disk_paths=("/mnt/nas05/data01/vincenzo/", "/"),
        ),
    ),
    wandb_logger = WandbLogger(log_model=False)  # Push it only on training end.

    # Overwrite config with wandb config (for sweep etc.)
    # And to make sure that we indeed pass all correct parameters.
    del config
    config = wandb.config

    # Print the full config
    print(dict(config))

    # Create dataset
    ds_train = load_dataset(
        config["data"]["train_path"], split=config["data"]["train_split"]
    )
    ds_valid = load_dataset(
        config["data"]["train_path"], split=config["data"]["val_split"]
    )

    # Transforms
    resize_func = Compose(
        [
            lambda x: custom_resize(x, tuple(config["model"]["input_size"])),
        ]
    )
    if config["data"]["use_augmentation"]:
        augm_before_resize = TimeWarpAugmenter(W=config["data"]["time_warp_w"])
        augm_after_resize = CustomSpecAugment(
            frequency_masking_para=config["data"]["frequency_masking_para"],
            time_masking_para=config["data"]["time_masking_para"],
            method=config["data"]["freq_mask_method"],
        )
    else:
        augm_before_resize = None
        augm_after_resize = None

    # Data Loader
    ds_train = EcallistoDatasetBinary(
        ds_train,
        resize_func=resize_func,
        normalization_transform=remove_background,
        augm_before_resize=augm_before_resize,
        augm_after_resize=augm_after_resize,
    )
    ds_valid = EcallistoDatasetBinary(
        ds_valid,
        resize_func=resize_func,
        normalization_transform=remove_background,
    )

    # Dataloader
    train_dataloader = DataLoader(
        ds_train,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        ds_valid,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    # Checkpoint to save the best model based on the lowest validation loss
    checkpoint_callback_f1 = ModelCheckpoint(
        monitor="val_barlow_loss",
        dirpath=wandb_logger.experiment.dir,
        filename="val_barlow_loss-{epoch:02d}-{step:05d}-{val_f1:.3f}",
        save_top_k=1,
        mode="min",
    )

    # Early stopping based on validation loss
    early_stopping_callback = EarlyStopping(
        monitor="val_barlow_loss",
        patience=3,  # It's 3 Epochs.
        verbose=True,
        mode="min",
    )

    # Setup Model
    cw = torch.tensor(ds_train.get_class_weights(), dtype=torch.float)
    model = ResNetBarlow(
        resnet_type=config["model"]["model_type"],
        batch_size=config["general"]["batch_size"],
        optimizer_name=config["model"]["optimizer_name"],
        learning_rate=config["model"]["lr"],
        augmentation_func=ds_train.augment_image,
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["general"]["max_epochs"],
        logger=wandb_logger,
        enable_progress_bar=False,
        val_check_interval=1.0,  # Every Epoch.
        callbacks=[checkpoint_callback_f1, early_stopping_callback],
    )

    # Train
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
