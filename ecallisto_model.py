# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryPrecision, BinaryF1Score, BinaryRecall
from torchvision import models
from collections import defaultdict
import wandb
from torchvision.transforms.functional import to_pil_image


class EcallistoBase(LightningModule):
    def __init__(self, n_classes, class_weights, unnormalize_img, batch_size):
        super().__init__()
        self.task = "binary" if n_classes == 1 else "multiclass"
        self.recall = BinaryRecall(task=self.task)
        self.precision = BinaryPrecision(task=self.task)
        self.f1_score = BinaryF1Score(task=self.task)
        self.confmat = ConfusionMatrix(task=self.task)
        self.class_weights = class_weights
        self.loss_function = F.cross_entropy
        self.batch_size = batch_size

        ## Test parameters
        self.fp_examples = []
        self.fn_examples = []
        self.results = defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
        self.unnormalize_img = unnormalize_img

    @staticmethod
    def _calculate_prediction(y_hat):
        probabilities = F.softmax(y_hat, dim=1).squeeze()
        preds = torch.where(probabilities[:, 1] > 0.5, 1, 0)
        return probabilities, preds

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        if self.class_weights is not None:
            loss = self.loss_function(y_hat, y, weight=self.class_weights.to(y.device))
        else:
            loss = self.loss_function(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, batch_size=self.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        _, preds = self._calculate_prediction(y_hat)
        # Update confusion matrix and other metrics
        self.confmat.update(preds, y)
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1_score.update(preds, y)

    def on_validation_epoch_end(self):
        # Calculate and log metrics
        pre = self.precision.compute()
        rec = self.recall.compute()
        f1 = self.f1_score.compute()

        # Log the computed metrics
        self.log(
            "val_precision",
            pre,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
        )
        self.log(
            "val_recall",
            rec,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_f1",
            f1,
            prog_bar=True,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

        # Calculate conf matrix
        confmat = self.confmat.compute()
        fig, ax = plt.subplots()
        sns.heatmap(confmat.cpu().numpy(), annot=True, fmt="g", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log the confusion matrix as an image to wandb
        self.logger.experiment.log(
            {"confusion_matrix": [wandb.Image(plt, caption="Val Confusion Matrix")]}
        )

        plt.close(fig)
        # Reset metrics for the next epoch
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.confmat.reset()

    def test_step(self, batch, batch_idx):
        images, labels, antennas, _ = batch
        y_hat = self(images)
        _, preds = self._calculate_prediction(y_hat)

        for img, label, pred, antenna in zip(images, labels, preds, antennas):
            key = (antenna, label)

            if label != pred:
                if label == 0:  # Assuming 0 is the negative class
                    self.results[key]["FP"] += 1
                    if len(self.fp_examples) < 15:
                        self.fp_examples.append((img, antenna, label))
                else:
                    if len(self.fn_examples) < 15:
                        self.fn_examples.append((img, antenna, label))

        # Calculate metrics
        y_hat = self(images)

        # Update confusion matrix and other metrics
        self.confmat.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1_score.update(preds, labels)

    def on_test_epoch_end(self):
        # Calculate and log metrics
        pre = self.precision.compute()
        rec = self.recall.compute()
        f1 = self.f1_score.compute()

        # Log the computed metrics
        self.log("test_precision", pre, prog_bar=True, batch_size=self.batch_size)
        self.log("test_recall", rec, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1", f1, prog_bar=True, batch_size=self.batch_size)

        # Calculate conf matrix
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
        # Reset metrics for the next epoch
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.confmat.reset()
        # Upload images
        fp_images = [
            wandb.Image(
                to_pil_image(self.unnormalize_img(img[0], img[1])),
                caption=f"Antenna: {img[1]}. Label: {img[2]}",
            )
            for img in self.fp_examples
        ]
        wandb.log({"False Positives": fp_images})

        # Convert and log false negative examples
        fn_images = [
            wandb.Image(
                to_pil_image(self.unnormalize_img(img[0], img[1])),
                caption=f"Antenna: {img[1]}. Label: {img[2]}",
            )
            for img in self.fn_examples
        ]
        wandb.log({"False Negatives": fn_images})

    def on_train_end(self):
        # Log the best model to wandb at the end of training
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            wandb.log_artifact(best_model_path, type="model", name="best_model")


class EfficientNet(EcallistoBase):
    def __init__(
        self,
        n_classes,
        class_weights=None,
        learnig_rate=None,
        unnormalize_img=None,
        dropout=None,
        batch_size=None,
        model_weights=None,
    ):
        super().__init__(
            n_classes=n_classes,
            class_weights=class_weights,
            unnormalize_img=unnormalize_img,
            batch_size=batch_size,
        )
        self.efficient_net = models.efficientnet_v2_s(
            weights=model_weights,
            dropout=dropout,
            num_classes=n_classes,
        )
        self.learnig_rate = learnig_rate

    def forward(self, x):
        # Convert grayscale image to 3-channel image if it's not already
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.efficient_net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learnig_rate)


class ResNet18(EcallistoBase):
    def __init__(
        self,
        n_classes,
        class_weights=None,
        learnig_rate=None,
        unnormalize_img=None,
        batch_size=None,
        model_weights=None,
    ):
        super().__init__(
            n_classes=n_classes,
            class_weights=class_weights,
            unnormalize_img=unnormalize_img,
            batch_size=batch_size,
        )
        self.resnet18 = models.resnet18(weights=model_weights, num_classes=n_classes)
        self.learnig_rate = learnig_rate

    def forward(self, x):
        # ResNet-18 expects 3-channel input, so ensure the input x is 3-channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet18(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learnig_rate)


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
        return image / 255.0

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
