# Modeling
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Recall, Precision, F1Score
from torchvision import models
import wandb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

RESNET_DICT = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet52": models.resnet50,
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
    ):
        super().__init__()
        self.recall = Recall(num_classes=n_classes, task="multiclass", average="macro")
        self.precision = Precision(
            num_classes=n_classes, task="multiclass", average="macro"
        )
        self.f1_score = F1Score(
            num_classes=n_classes, task="multiclass", average="macro"
        )
        self.confmat = ConfusionMatrix(num_classes=n_classes, task="multiclass")
        self.class_weights = class_weights
        self.loss_function = F.cross_entropy
        self.batch_size = batch_size

        # Optimizer
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_prediction(y_hat):
        probabilities = F.softmax(y_hat, dim=1).squeeze()
        preds = torch.where(probabilities[:, 1] > 0.5, 1, 0)
        return probabilities, preds

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

    def _loss(self, y_hat, y):
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

        # Log the confusion matrix
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

    def on_train_end(self):
        # Log the best model to wandb at the end of training
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            wandb.log_artifact(best_model_path, type="model", name="best_model")

    def configure_optimizers(self):
        return OPTIMIZERS[self.optimizer_name](
            params=self.parameters(), lr=self.learning_rate, eps=1e-6
        )


class ResNet(EcallistoBase):
    def __init__(
        self,
        n_classes,
        resnet_type,
        optimizer_name,
        learning_rate,
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
        )
        self.resnet = RESNET_DICT[resnet_type](
            weights=model_weights, num_classes=n_classes
        )

    def forward(self, x):
        # ResNet-18 expects 3-channel input, so ensure the input x is 3-channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)


def create_normalize_function(antenna_stats, simple):
    def normalize(image, antenna):
        # Retrieve the statistics for the given antenna
        stats = antenna_stats[antenna]

        # Apply normalization (Assuming image is a torch.Tensor)
        normalized_image = (image - stats["min"]) / (stats["max"] - stats["min"])
        # normalized_image = (2 * ((image - stats["min"]) / (stats["max"] - stats["min"])) - 1)
        normalized_image = (normalized_image - stats["mean"]) / stats["std"]

        return normalized_image

    def simple_normalize(image, antenna):
        return image / 254.0

    return simple_normalize if simple else normalize


def create_unnormalize_function(antenna_stats):
    def unnormalize(normalized_image, antenna):
        # Retrieve the statistics for the given antenna
        stats = antenna_stats[antenna]

        # Reverse the normalization
        unnormalized_image = normalized_image * stats["std"] + stats["mean"]
        unnormalized_image = (
            unnormalized_image * (stats["max"] - stats["min"]) + stats["min"]
        )

        return unnormalized_image.to(torch.uint8)

    return unnormalize
