# Visualization
import numpy as np

# Modeling
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize

import wandb
from ecallisto_dataset import EcallistoData
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

    ds = load_dataset("i4ds/radio-sunburst-ecallisto")

    dd = DatasetDict()
    dd["train"] = ds["train"]
    dd["test"] = ds["test"]
    dd["validation"] = ds["validation"]

    normalize = Normalize(mean=0.5721, std=0.1100)
    size = (224, 244)

    _transforms = Compose([Resize(size), normalize])

    ds_train = EcallistoData(dd["train"], transform=_transforms)
    ds_valid = EcallistoData(dd["validation"], transform=_transforms)
    ds_test = EcallistoData(dd["test"], return_all_columns=True)

    BATCH_SIZE = 32

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

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(dd["train"]["label"]),
        y=dd["train"]["label"],
    )

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
        len(np.unique(dd["train"]["label"])), torch.tensor(cw, dtype=torch.float)
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
