# Modeling
import argparse
import torch
import yaml
from datasets import DatasetDict, load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose, Resize

import wandb
from ecallisto_dataset import (
    randomly_reduce_class_samples,
    EcallistoDatasetBinary,
)
from ecallisto_model import (
    ResNet,
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
    wandb_logger = WandbLogger(log_model=False)  # Push it only on training end.

    # Overwrite config with wandb config (for sweep etc.)
    del config
    config = wandb.config

    # Print the full config
    print(dict(config))

    # Create dataset
    ds_train = load_dataset(config["data"]["train_path"], split="train")
    ds_val = load_dataset(config["data"]["train_path"], split="validation")

    if config["data"]["reduce_non_burst"]:
        ds_train = randomly_reduce_class_samples(
            ds_train,
            config["data"]["train_class_to_reduce"],
            config["data"]["reduction_fraction"],
        )
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
    normalize_transform = create_normalize_function(
        antenna_stats=antenna_stats, simple=config["data"]["simple_normalization"]
    )

    # Data Loader
    ds_train = EcallistoDatasetBinary(
        ds_train,
        base_transform=base_transform,
        data_augm_transform=data_augm_transform,
        normalization_transform=normalize_transform,
    )
    ds_valid = EcallistoDatasetBinary(
        ds_val,
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
        monitor="val_loss",
        dirpath=wandb_logger.experiment.dir,
        filename="loss-{epoch:02d}-{step:05d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Setup Model
    cw = torch.tensor(ds_train.get_class_weights(), dtype=torch.float)
    unnormalize_img = create_unnormalize_function(antenna_stats)
    model = ResNet(
        n_classes=2,  # Binary
        resnet_type=config["model"]["model_type"],
        class_weights=(cw if config["general"]["use_class_weights"] else None),
        batch_size=config["general"]["batch_size"],
        learnig_rate=config["model"]["lr"],
        unnormalize_img=unnormalize_img,
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["general"]["max_epochs"],
        logger=wandb_logger,
        enable_progress_bar=False,
        val_check_interval=200,
    )

    # Train
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    ## Evaluate
    # Create test set
    ds_test = load_dataset(config["data"]["test_path"], split="test")

    ds_test = EcallistoDatasetBinary(
        ds_test,
        base_transform=base_transform,
        normalization_transform=normalize_transform,
    )
    # Create dataloader
    test_dataloader = DataLoader(
        ds_test,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=True,  # To randomly log images
        persistent_workers=False,
    )
    trainer.test(model, test_dataloader, ckpt_path="best")
