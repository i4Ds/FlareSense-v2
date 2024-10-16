# Modeling
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose

import wandb
from ecallisto_dataset import (
    CustomSpecAugment,
    EcallistoBarlowDataset,
    TimeWarpAugmenter,
    custom_resize,
    remove_background,
)

RESNET_DICT = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, batch_size, lambda_bt=5e-3):
    """
    Computes the Barlow Twins loss between two sets of embeddings.

    Args:
        z1 (torch.Tensor): Embeddings from the first view.
        z2 (torch.Tensor): Embeddings from the second view.
        lambda_bt (float): Weighting factor for the off-diagonal loss.

    Returns:
        torch.Tensor: The Barlow Twins loss.
    """
    # N x D, where N is the batch size and D is output dim of projection head
    z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
    z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

    cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size

    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(cross_corr).pow_(2).sum()

    # Normalize by dimensionality
    D = z1.size(1)
    loss = (on_diag + lambda_bt * off_diag) / D

    return loss


class ResNetBarlow(LightningModule):
    def __init__(
        self,
        resnet_type,
        batch_size,
        optimizer_name,
        learning_rate,
        augmentation_func,
        warmup_lr,
        model_weights: str = None,
        projection_dim: int = 8192,
        projection_hidden_dim: int = 8192,
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
            warmup_lr (int): Epochs for learning rate warmup.
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
        self.warmup_lr = warmup_lr

        # Get model
        resnet_cls = RESNET_DICT.get(resnet_type)

        ### BARLOW STUFF
        # Initialize ResNet without the final classification layer
        self.encoder = resnet_cls(weights=model_weights)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection head as per Barlow Twins
        self.projector = nn.Sequential(
            nn.Linear(in_features, projection_hidden_dim, bias=True),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim, bias=False),
        )
        # Hyperparameters for Barlow Twins loss
        self.lambda_bt = lambda_bt

        ## Track some stuff
        self.val_embeddings = []
        self.val_labels = []

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
        x, _, y, _, _ = batch

        # Pass through the model to get embeddings
        z1 = self(x)

        self.val_embeddings.append(z1.cpu())
        self.val_labels.append(y.cpu())

    def on_validation_epoch_end(self):
        all_embeddings = torch.cat(self.val_embeddings, dim=0).numpy()
        all_labels = torch.cat(self.val_labels, dim=0).numpy().astype(int)

        # Check that at least two classes are there
        if len(np.unique(all_labels)) < 2:
            # Create 2 dummy classes
            all_labels = np.random.choice([0, 1], size=len(all_labels))

        # Clear the lists for the next epoch
        self.val_embeddings.clear()
        self.val_labels.clear()

        # Define the classifier
        clf = LogisticRegression(max_iter=1000)

        # Use Stratified K-Fold to maintain label distribution across folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Compute cross-validated F1 scores
        f1_scores = cross_val_score(
            clf,
            all_embeddings,
            all_labels,
            cv=skf,
            scoring=make_scorer(f1_score, average="binary"),
        )

        # Calculate the mean F1 score
        mean_f1 = np.mean(f1_scores)

        self.log("val_f1", mean_f1, prog_bar=True)

    def training_step(self, batch, batch_idx):
        xt1, xt2, _, _, _ = batch
        # Pass through the model to get embeddings
        z1 = self(xt1)
        z2 = self(xt2)

        # Barlow Twins loss
        barlow_loss = barlow_twins_loss(
            z1, z2, batch_size=self.batch_size, lambda_bt=self.lambda_bt
        )

        # Logging
        self.log("barlow_loss", barlow_loss, on_step=True, batch_size=self.batch_size)

        # Return the loss
        return barlow_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, total_iters=self.warmup_lr
            ),
        }

        return [optimizer], [scheduler]


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
        project="FlareSense-Barlow-Twins",
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
    augm_before_resize = TimeWarpAugmenter(W=config["data"]["time_warp_w"])
    augm_after_resize = CustomSpecAugment(
        frequency_masking_para=config["data"]["frequency_masking_para"],
        time_masking_para=config["data"]["time_masking_para"],
        method=config["data"]["freq_mask_method"],
    )

    # Data Loader
    ds_train = EcallistoBarlowDataset(
        ds_train,
        resize_func=resize_func,
        normalization_transform=remove_background,
        augm_before_resize=augm_before_resize,
        augm_after_resize=augm_after_resize,
        delete_cache_after_run=False,
    )
    ds_valid = EcallistoBarlowDataset(
        ds_valid,
        resize_func=resize_func,
        normalization_transform=remove_background,
        delete_cache_after_run=False,
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

    # Checkpoint to save the best model based on the lowest validation f1
    checkpoint_callback_f1 = ModelCheckpoint(
        monitor="val_f1",
        dirpath=wandb_logger.experiment.dir,
        filename="val_f1-{epoch:02d}-{step:05d}-{val_f1:.3f}",
        save_top_k=1,
        mode="max",
    )

    # Early stopping based on validation loss
    early_stopping_callback = EarlyStopping(
        monitor="val_f1",
        patience=3,  # It's 3 Epochs.
        verbose=True,
        mode="max",
    )

    # Learning rate logger
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup Model
    model = ResNetBarlow(
        resnet_type=config["model"]["model_type"],
        batch_size=config["general"]["batch_size"],
        optimizer_name=config["model"]["optimizer_name"],
        learning_rate=config["model"]["lr"],
        warmup_lr=config["model"]["warmup_lr"],
        projection_hidden_dim=config["model"]["projection_hidden_dim"],
        projection_dim=config["model"]["projection_dim"],
        augmentation_func=ds_train.augment_image,
        lambda_bt=config["model"]["lambda_bt"],
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["general"]["max_epochs"],
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback_f1, early_stopping_callback, lr_monitor],
    )

    # Train
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
