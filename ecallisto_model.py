# Modeling
# Visualization
from collections import defaultdict
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1Score, Precision, Recall
from torchvision import models
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR

RESNET_DICT = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
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
        max_epochs,
        warmup_epochs,
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
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

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
        targets = targets.float()
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
        x, y, antennas, _ = batch  # Antennas is the third position
        y_hat = self(x)

        # Loss
        loss = self._loss(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
        )

        # Store predictions, ground truths, and antennas for group-wise metrics
        self.validation_outputs.append(
            {
                "y_hat": y_hat,
                "y": y,
                "antennas": antennas,
            }
        )

    def on_validation_epoch_start(self):
        # Reset the validation outputs storage
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        # Group outputs by antenna
        grouped_metrics = defaultdict(lambda: {"y_hat": [], "y": []})
        for output in self.validation_outputs:
            for i, antenna in enumerate(output["antennas"]):
                grouped_metrics[antenna]["y_hat"].append(output["y_hat"][i])
                grouped_metrics[antenna]["y"].append(output["y"][i])

        # Calculate metrics per antenna
        antenna_f1_scores = []
        for antenna, values in grouped_metrics.items():
            y_hat_group = torch.cat(values["y_hat"])
            y_group = torch.cat(values["y"])
            f1 = self.f1_score(y_hat_group, y_group)
            antenna_f1_scores.append(f1)
            self.log(f"val_f1_{antenna}", f1, prog_bar=False)

        # Log averaged F1 score across antennas
        avg_f1 = torch.mean(torch.tensor(antenna_f1_scores))
        self.log("val_avg_f1", avg_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, antennas, _ = batch  # Antennas is the third position
        y_hat = self(x)

        # Loss
        loss = self._loss(y_hat, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
        )

        # Store predictions, ground truths, and antennas for group-wise metrics
        self.test_outputs.append(
            {
                "y_hat": y_hat,
                "y": y,
                "antennas": antennas,
            }
        )

    def on_test_epoch_start(self):
        # Reset the test outputs storage
        self.test_outputs = []

    def on_test_epoch_end(self):
        # Group outputs by antenna
        grouped_metrics = defaultdict(lambda: {"y_hat": [], "y": []})
        for output in self.test_outputs:
            for i, antenna in enumerate(output["antennas"]):
                grouped_metrics[antenna]["y_hat"].append(output["y_hat"][i])
                grouped_metrics[antenna]["y"].append(output["y"][i])

        # Calculate metrics per antenna
        antenna_precision_scores = []
        antenna_recall_scores = []
        antenna_f1_scores = []

        # Save also predictions
        y_hat = []
        y = []

        for antenna, values in grouped_metrics.items():
            y_hat_group = torch.cat(values["y_hat"])
            y_group = torch.cat(values["y"])
            y_hat.append(y_hat_group)
            y.append(y_group)

            precision = self.precision(y_hat_group, y_group)
            recall = self.recall(y_hat_group, y_group)
            f1 = self.f1_score(y_hat_group, y_group)

            antenna_precision_scores.append(precision)
            antenna_recall_scores.append(recall)
            antenna_f1_scores.append(f1)

            # Log metrics per antenna
            self.log(f"test_precision_{antenna}", precision, prog_bar=False)
            self.log(f"test_recall_{antenna}", recall, prog_bar=False)
            self.log(f"test_f1_{antenna}", f1, prog_bar=False)

        # Log averaged metrics across antennas
        avg_precision = torch.mean(torch.tensor(antenna_precision_scores))
        avg_recall = torch.mean(torch.tensor(antenna_recall_scores))
        avg_f1 = torch.mean(torch.tensor(antenna_f1_scores))

        self.log("test_avg_precision", avg_precision, prog_bar=True)
        self.log("test_avg_recall", avg_recall, prog_bar=True)
        self.log("test_avg_f1", avg_f1, prog_bar=True)

        # Also add the weighted f1, precision, and recall
        self.log("test_f1", self.f1_score(torch.cat(y_hat), torch.cat(y)))
        self.log("test_precision", self.precision(torch.cat(y_hat), torch.cat(y)))
        self.log("test_recall", self.recall(torch.cat(y_hat), torch.cat(y)))

    def on_train_end(self):
        # Check if the run is not part of a sweep
        if wandb.run.sweep_id is None and not self.trainer.sanity_checking:
            # Save the model manually
            model_path = "final_model.pth"
            torch.save(self.state_dict(), model_path)

            # Log the saved model to wandb
            artifact = wandb.Artifact("final_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def configure_optimizers(self):
        if self.warmup_epochs >= self.max_epochs:
            raise ValueError(
                "Warm-up epochs should be less than the total number of epochs."
            )
        # Initialize optimizer
        optimizer = OPTIMIZERS[self.optimizer_name](
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Warm-up phase
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Starting LR as a fraction of the initial LR
            end_factor=1.0,  # End of warm-up phase
            total_iters=self.warmup_epochs,
        )

        # Decay phase
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.max_epochs - self.warmup_epochs,
        )

        # Combine warm-up and decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.warmup_epochs],
        )

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  # Frequency of stepping
            },
        }


class GrayScaleResNet(EcallistoBase):
    def __init__(
        self,
        n_classes,
        resnet_type,
        optimizer_name,
        learning_rate,
        max_epochs,
        warmup_epochs,
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
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        self.resnet = RESNET_DICT[resnet_type](
            weights=model_weights, num_classes=n_classes
        )
        # Compile
        self.resnet = torch.compile(self.resnet)

    def forward(self, x):
        # Convert the input grayscale image to 3 channels
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.resnet(x)


if __name__ == "__main__":
    x = torch.load("img.torch")
    print(x.shape)
    model = GrayScaleResNet(2, "resnet18", "adam", 1, 5)
    print(model(x.unsqueeze(0)))
