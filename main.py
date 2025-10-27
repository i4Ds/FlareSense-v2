# Modeling
import argparse
import os

import torch
import yaml
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
import wandb
from ecallisto_dataset import (
    CustomSpecAugment,
    EcallistoDatasetBinary,
    TimeWarpAugmenter,
    custom_resize,
    normal_resize,
    filter_antennas,
    randomly_reduce_class_samples,
    remove_background,
    clip_tensor,
)
from ecallisto_model import GrayScaleResNet
from pytorch_lightning.callbacks import LearningRateMonitor

if __name__ == "__main__":
    print(f"PyTorch version {torch.__version__}")
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_id = os.environ.get("SLURM_JOB_GPUS")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(
            f"GPU is available: {device_name} (Device ID: {device_id}). I have {os.cpu_count()} cores available."
        )
        device = "cuda"
    else:
        print("GPU is not available.")
        device = "cpu"

    # Mixed precision and performance optimizations
    torch.set_float32_matmul_precision("high")

    # Enable cuDNN benchmarking for additional speedup
    # This finds the best convolution algorithms for your specific hardware
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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
    # Train can be a list of datasets, so we need to iterate over them and then concatenate them
    ds_train = []
    for ds, split in zip(config["data"]["train_path"], config["data"]["train_split"]):
        ds_train.append(load_dataset(ds, split=split))
    ds_train = concatenate_datasets(ds_train)

    ds_valid = load_dataset(
        config["data"]["val_path"],
        split=config["data"]["val_split"],
    )

    if config["data"]["reduce_non_burst"]:
        ds_train = randomly_reduce_class_samples(
            ds_train,
            config["data"]["train_class_to_reduce"],
            config["data"]["reduction_fraction"],
        )

    # Filter by certain antennas
    if len(config["data"]["antennas_train"]) > 0:
        ds_train = filter_antennas(ds_train, config["data"]["antennas_train"])
    if len(config["data"]["antennas_val"]) > 0:
        ds_valid = filter_antennas(ds_valid, config["data"]["antennas_val"])

    # Transforms
    if config["data"]["custom_resize"]:
        resize_func = Compose(
            [
                lambda x: custom_resize(x, tuple(config["model"]["input_size"])),
            ]
        )
    else:
        resize_func = Compose(
            [
                lambda x: normal_resize(x, tuple(config["model"]["input_size"])),
            ]
        )

    if config["data"]["use_augmentation"]:
        augm_before_resize = TimeWarpAugmenter(W=config["data"]["time_warp_w"])
        if config["data"]["clip_to_range"]:
            augm_after_resize = Compose(
                [
                    lambda x: clip_tensor(x, (0, 16)),
                    CustomSpecAugment(
                        frequency_masking_para=config["data"]["frequency_masking_para"],
                        time_masking_para=config["data"]["time_masking_para"],
                        method=config["data"]["freq_mask_method"],
                    ),
                ]
            )
        else:
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
        label_name=config["data"]["train_label_name"],
        resize_func=resize_func,
        normalization_transform=remove_background,
        augm_before_resize=augm_before_resize,
        augm_after_resize=augm_after_resize,
    )
    ds_valid = EcallistoDatasetBinary(
        ds_valid,
        label_name=config["data"]["val_label_name"],
        resize_func=resize_func,
        normalization_transform=remove_background,
    )

    # Incase we want to get a weighted random sampler (oversample rare classes)
    if config["general"]["use_random_sampler"]:
        sample_weights = ds_train.get_sample_weights()
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
        prefetch_factor=6,
        pin_memory=True,
        shuffle=False if sampler is not None else True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        ds_valid,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        prefetch_factor=6,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup Model
    cw = torch.tensor(ds_train.get_class_weights(), dtype=torch.float)
    model = GrayScaleResNet(
        n_classes=1,  # Binary
        resnet_type=config["model"]["model_type"],
        class_weights=(cw if config["general"]["use_class_weights"] else None),
        batch_size=config["general"]["batch_size"],
        optimizer_name=config["model"]["optimizer_name"],
        max_epochs=config["general"]["max_epochs"],
        warmup_epochs=config["model"]["warmup_epochs"],
        learning_rate=config["model"]["lr"],
        weight_decay=config["model"]["weight_decay"],
        label_smoothing=config["model"]["label_smoothing"],
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config["general"]["max_epochs"],
        logger=wandb_logger,
        enable_progress_bar=False,
        val_check_interval=1.0,  # Every Epoch.
        # callbacks=[checkpoint_callback_f1, early_stopping_callback],
        callbacks=[lr_monitor],
        precision="16-mixed",  # Enable automatic mixed precision training
    )

    # Train
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    ## Evaluate
    # Create test set
    ds_test = load_dataset(
        config["data"]["test_path"], split=config["data"]["test_split"]
    )

    # Filter
    if len(config["data"]["antennas_test"]) > 0:
        ds_test = filter_antennas(ds_test, config["data"]["antennas_test"])

    ds_test = EcallistoDatasetBinary(
        ds_test,
        label_name=config["data"]["test_label_name"],
        resize_func=resize_func,
        normalization_transform=remove_background,
    )
    # Create dataloader
    test_dataloader = DataLoader(
        ds_test,
        batch_size=config["general"]["batch_size"],
        num_workers=8,
        shuffle=False,
        persistent_workers=False,
    )
    trainer.test(model, test_dataloader)
