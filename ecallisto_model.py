# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
from torchvision import models

import wandb


class EcallistoBase(LightningModule):
    def __init__(self, n_classes, class_weights):
        super().__init__()
        self.accuracy = Accuracy(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.f1_score = F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.class_weights = class_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.confmat.update(preds, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True)
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
