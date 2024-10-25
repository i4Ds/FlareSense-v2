# Modeling
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1Score, Precision, Recall
from torchvision import models

import wandb

RESNET_DICT = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


class EcallistoBase(LightningModule):
    def __init__(
        self,
        n_classes,
        class_weights,
        batch_size,
        optimizer_name,
        learning_rate,
        weight_decay,
        label_smoothing=0.0,
    ):
        super().__init__()
        assert n_classes >= 1, "You have 0 Classes?"
        if n_classes == 1:
            self.recall = Recall(task="binary")
            self.precision = Precision(task="binary")
            self.f1_score = F1Score(task="binary")
            self.confmat = ConfusionMatrix(task="binary")
        else:
            self.recall = Recall(
                num_classes=n_classes, task="multiclass", average="macro"
            )
            self.precision = Precision(
                num_classes=n_classes, task="multiclass", average="macro"
            )
            self.f1_score = F1Score(
                num_classes=n_classes, task="multiclass", average="macro"
            )
            self.confmat = ConfusionMatrix(num_classes=n_classes, task="multiclass")
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        """
        self.loss_function = (
            F.cross_entropy if n_classes > 1 else F.binary_cross_entropy_with_logits
        )
        """
        # Simplyfied pipeline, we are only looking at binary currently.
        self.loss_function = F.binary_cross_entropy_with_logits
        self.batch_size = batch_size

        # Optimizer
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # List to save test results
        self.x_test = []

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=False,
        )
        return loss

    @staticmethod
    def apply_label_smoothing(targets, smoothing):
        """
        Apply label smoothing to binary targets.

        Args:
            targets (torch.Tensor): Tensor of shape (N,) with binary labels (0 or 1).
            smoothing (float): Smoothing factor between 0 and 1.

        Returns:
            torch.Tensor: Smoothed targets.
        """
        # Ensure targets are floats
        targets = targets.float().to(targets.device)
        # Apply label smoothing
        smoothed_targets = targets * (1.0 - smoothing) + (1.0 - targets) * smoothing
        return smoothed_targets

    def _loss(self, y_hat, y):
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            y = self.apply_label_smoothing(y, self.label_smoothing)
        if self.class_weights is not None:
            loss = self.loss_function(y_hat, y, weight=self.class_weights.to(y.device))
        else:
            loss = self.loss_function(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        # Metrics
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1_score(y_hat, y)
        # Loss
        loss = self._loss(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

        # Log the computed metrics
        self.log(
            "val_precision",
            self.precision,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_recall",
            self.recall,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_f1",
            self.f1_score,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

    def test_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)

        # Metrics
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1_score(y_hat, y)
        self.confmat(y_hat, y)

        # Loss
        loss = self._loss(y_hat, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

        # Log the computed metrics
        self.log(
            "test_precision",
            self.precision,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "test_recall",
            self.recall,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "test_f1",
            self.f1_score,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

    def on_test_epoch_end(self):
        # Compute the confusion matrix
        confmat = self.confmat.compute()
        fig, ax = plt.subplots()
        sns.heatmap(confmat.cpu().numpy(), annot=True, fmt="g", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log the confusion matrix as an image to wandb
        self.logger.experiment.log(
            {"confusion_matrix": [wandb.Image(plt, caption="Test Confusion Matrix")]}
        )

        plt.close(fig)

        # Reset confusion matrix for the next epoch
        self.confmat.reset()

    def on_train_end(self):
        # Log the best model to wandb at the end of training
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            wandb.log_artifact(best_model_path, type="model", name="best_model")

    def configure_optimizers(self):
        return OPTIMIZERS[self.optimizer_name](
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class GrayScaleResNet(EcallistoBase):
    def __init__(
        self,
        n_classes,
        resnet_type,
        optimizer_name,
        learning_rate,
        weight_decay=None,
        label_smoothing=None,
        class_weights=None,
        batch_size=None,
        model_weights=None,
    ):
        super().__init__(
            n_classes=n_classes,
            class_weights=class_weights,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        self.resnet = RESNET_DICT[resnet_type](
            weights=model_weights, num_classes=n_classes
        )

    def forward(self, x):
        # Convert the input grayscale image to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)
