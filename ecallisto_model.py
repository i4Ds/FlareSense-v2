# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Recall, Precision
from torchvision import models
from collections import defaultdict
import wandb
from torchvision.transforms.functional import to_pil_image


class EcallistoBase(LightningModule):
    def __init__(self, n_classes, class_weights, unnormalize_img):
        super().__init__()
        self.recall = Recall(task="multiclass", num_classes=n_classes)
        self.precision = Precision(task="multiclass", num_classes=n_classes)
        self.f1_score = F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.class_weights = class_weights

        ## Test parameters
        self.fp_examples = []
        self.fn_examples = []
        self.results = defaultdict(lambda: {"correct": 0, "total": 0})
        self.unnormalize_img = unnormalize_img

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        if self.class_weights is not None:
            loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(y.device))
        else:
            loss = F.cross_entropy(y_hat, y)
        # logs metrics for each training_step - [default:True],
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.confmat.update(preds, y)
        pre = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        self.log("val_precision", pre, prog_bar=True, logger=True)
        self.log("val_recall", rec, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        confmat = self.confmat.compute()
        fig, ax = plt.subplots()
        sns.heatmap(confmat.cpu().numpy(), annot=True, fmt="g", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log the confusion matrix as an image to wandb
        self.logger.experiment.log(
            {"confusion_matrix": [wandb.Image(plt, caption="Confusion Matrix")]}
        )

        plt.close(fig)

        # Reset the confusion matrix for the next epoch
        self.confmat.reset()

    def test_step(self, batch, batch_idx):
        images, labels, antennas, types = batch
        outputs = self(images)
        _, preds = torch.max(outputs, dim=1)

        for label, pred, antenna, typ in zip(labels, preds, antennas, types):
            key = (antenna, typ)

            if label == pred:
                if label == 1:  # Assuming 1 is the positive class
                    self.results[key]["TP"] += 1
                else:
                    self.results[key]["TN"] += 1
            else:
                if label == 0:  # Assuming 0 is the negative class
                    self.results[key]["FP"] += 1
                    if len(self.fp_examples) < 15:
                        self.fp_examples.append((images, antenna, typ))
                else:
                    self.results[key]["FN"] += 1
                    if len(self.fn_examples) < 15:
                        self.fn_examples.append((images, antenna, typ))

    def on_test_epoch_end(self, outputs):
        # Initialize a wandb Table
        metrics_table = wandb.Table(
            columns=["Antenna", "Type", "Precision", "Recall", "F1"]
        )

        overall_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for key, result in self.results.items():
            precision = (
                result["TP"] / (result["TP"] + result["FP"])
                if (result["TP"] + result["FP"]) > 0
                else 0
            )
            recall = (
                result["TP"] / (result["TP"] + result["FN"])
                if (result["TP"] + result["FN"]) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Add data to the table
            metrics_table.add_data(key[0], key[1], precision, recall, f1)

            # Update overall metrics
            for metric in overall_metrics.keys():
                overall_metrics[metric] += result[metric]

        # Calculate overall metrics
        overall_precision = (
            overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FP"])
            if (overall_metrics["TP"] + overall_metrics["FP"]) > 0
            else 0
        )
        overall_recall = (
            overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FN"])
            if (overall_metrics["TP"] + overall_metrics["FN"]) > 0
            else 0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0
        )

        # Log the table to wandb
        wandb.log({"Metrics by Antenna/Type": metrics_table})

        # Optionally, log overall metrics as a separate entry or include them in the table as well
        wandb.log(
            {
                "Overall Precision": overall_precision,
                "Overall Recall": overall_recall,
                "Overall F1": overall_f1,
            }
        )

        # Upload images
        fp_images = [
            wandb.Image(
                to_pil_image(self.unnormalize_img(img[0], img[1])),
                caption=f"Antenna: {img[1]}. Type: {img[2]}",
            )
            for img in self.fp_examples
        ]
        wandb.log({"False Positives": fp_images})

        # Convert and log false negative examples
        fn_images = [
            wandb.Image(
                to_pil_image(self.unnormalize_img(img[0], img[1])),
                caption=f"Antenna: {img[1]}. Type: {img[2]}",
            )
            for img in self.fn_examples
        ]
        wandb.log({"False Negatives": fn_images})


class EfficientNet(EcallistoBase):
    def __init__(self, n_classes, class_weights, learnig_rate, model_weights=None):
        super().__init__(n_classes=n_classes, class_weights=class_weights)
        self.efficient_net = models.efficientnet_v2_s(weights=model_weights)
        self.learnig_rate = learnig_rate
        # Dynamically obtain the in_features from the current classifier layer
        in_features = self.efficient_net.classifier[1].in_features

        # Adapt the classifier layer to the number of output classes
        self.efficient_net.classifier[1] = nn.Linear(
            in_features=in_features, out_features=n_classes, bias=True
        )

    def forward(self, x):
        # Convert grayscale image to 3-channel image if it's not already
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.efficient_net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learnig_rate)


class ResNet18(EcallistoBase):
    def __init__(self, n_classes, class_weights, model_weights=None):
        super().__init__(n_classes=n_classes, class_weights=class_weights)
        self.resnet18 = models.resnet18(weights=model_weights)

        # Replace the final fully connected layer with a new one adapted to the number of classes
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(
            in_features=in_features, out_features=n_classes, bias=True
        )

    def forward(self, x):
        # ResNet-18 expects 3-channel input, so ensure the input x is 3-channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet18(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def create_normalize_function(antenna_stats):
    def normalize(image, antenna):
        # Retrieve the statistics for the given antenna
        stats = antenna_stats[antenna]

        # Apply normalization (Assuming image is a torch.Tensor)
        normalized_image = (image - stats["min"]) / (stats["max"] - stats["min"])
        normalized_image = (normalized_image - stats["mean"]) / stats["std"]

        return normalized_image

    return normalize


def create_unnormalize_function(antenna_stats):
    def unnormalize(normalized_image, antenna):
        # Retrieve the statistics for the given antenna
        stats = antenna_stats[antenna]

        # Reverse the normalization
        unnormalized_image = normalized_image * stats["std"] + stats["mean"]
        unnormalized_image = (
            unnormalized_image * (stats["max"] - stats["min"]) + stats["min"]
        )

        return unnormalized_image

    return unnormalize
