# Modeling
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose, Normalize, Resize

import wandb
from ecallisto_dataset import EcallistoData, randomly_reduce_class_samples
from ecallisto_model import EfficientNet

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

    # Create dataset
    ds = load_dataset("i4ds/radio-sunburst-ecallisto")

    dd = DatasetDict()
    dd["train"] = randomly_reduce_class_samples(ds["train"], 0, 0.2)
    dd["test"] = ds["test"]
    dd["validation"] = ds["validation"]

    # Define augmentation
    normalize = Normalize(mean=0.5721, std=0.1100)  # Calculated from the train dataset
    size = (224, 244)

    # Transforms
    base_transform = Compose(
        [
            Resize(size),  # Resize the image
        ]
    )
    data_augm_transform = Compose(
        [
            FrequencyMasking(freq_mask_param=30),  # Apply frequency masking
            TimeMasking(time_mask_param=30),  # Apply time masking
        ]
    )

    normalize_transform = Compose(
        [
            normalize,  # Normalize the image
        ]
    )

    # Data Loader
    ds_train = EcallistoData(
        dd["train"],
        binary_class=False,
        base_transform=base_transform,
        data_augm_transform=data_augm_transform,
        normalization_transform=normalize_transform,
    )
    ds_valid = EcallistoData(
        dd["validation"],
        binary_class=False,
        base_transform=base_transform,
        normalization_transform=normalize_transform,
    )
    ds_test = EcallistoData(
        dd["test"],
        binary_class=False,
        base_transform=base_transform,
        normalization_transform=normalize_transform,
        return_all_columns=True,
    )

    # Batch size
    BATCH_SIZE = 32

    # Create Data loader
    class_weights = ds_train.get_sample_weights()
    sampler = WeightedRandomSampler(
        class_weights, num_samples=len(class_weights), replacement=True
    )

    train_dataloader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        num_workers=14,
        shuffle=True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        num_workers=14,
        shuffle=False,
        persistent_workers=True,
    )

    cw = ds_train.get_class_weights()

    wandb.init(entity="vincenzo-timmel")
    wandb_logger = WandbLogger(log_model="all")

    # Checkpoint to save the best model based on the lowest validation loss
    checkpoint_callback_loss = ModelCheckpoint(
        monitor="val_loss",
        dirpath=wandb_logger.experiment.dir,
        filename="efficientnet-loss-{epoch:02d}-{step:05d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Checkpoint to save the best model based on the highest F1 score
    checkpoint_callback_f1 = ModelCheckpoint(
        monitor="val_f1",
        dirpath=wandb_logger.experiment.dir,
        filename="efficientnet-f1-{epoch:02d}-{step:05d}-{val_f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    model = EfficientNet(
        len(ds_train.get_labels()), torch.tensor(cw, dtype=torch.float)
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_f1],
        val_check_interval=200,
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
