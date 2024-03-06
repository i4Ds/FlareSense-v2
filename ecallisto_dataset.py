import random

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def to_torch_tensor(example):
    # Convert the PIL image to a tensor
    example["image"] = pil_to_tensor(example["image"])
    return example


def scale(example, max_value=255):
    example["image"] = torch.div(example["image"], max_value)
    return example


def preprocess(example):
    example = to_torch_tensor(example)
    example = scale(example)
    return example


def randomly_reduce_class_samples(dataset, target_class, fraction_to_keep):
    """
    Randomly reduce the number of samples for a specific class in a dataset.

    :param dataset: The dataset to process (assumed to be a Hugging Face dataset).
    :param target_class: The class label of the class to reduce.
    :param fraction_to_keep: Fraction of the target class samples to keep (value between 0 and 1).
    :return: A new dataset with reduced samples for the target class.
    """

    # Get all indices for the target class and other classes
    target_class_indices = [
        i for i, label in enumerate(dataset["label"]) if label == target_class
    ]
    other_class_indices = [
        i for i, label in enumerate(dataset["label"]) if label != target_class
    ]

    # Determine the number of target class samples to keep
    num_samples_to_keep = int(len(target_class_indices) * fraction_to_keep)

    # Randomly select the target class indices to keep
    selected_target_class_indices = np.random.choice(
        target_class_indices, num_samples_to_keep, replace=False
    )

    # Combine the selected target class indices with other class indices
    final_indices = list(selected_target_class_indices) + other_class_indices

    # Create and return the balanced dataset
    balanced_dataset = dataset.select(final_indices)

    return balanced_dataset


# Dataset
class EcallistoData(Dataset):
    def __init__(
        self,
        dataset,
        base_transform,
        normalization_transform,
        binary_class=False,
        data_augm_transform=None,
        return_all_columns=False,
    ):
        super().__init__()
        self.data = dataset
        self.binary_class = binary_class
        self.data_augm_transform = data_augm_transform
        self.base_transform = base_transform
        self.normalization_transform = normalization_transform
        self.return_all_columns = return_all_columns
        self.class_weights = self.get_class_weights()

    @staticmethod
    def to_torch_tensor(example):
        # Convert the example to a torch tensor.
        example["image"] = pil_to_tensor(example["image"])
        example["label"] = torch.tensor(example["label"])
        return example

    @staticmethod
    def scale(img, max_value=255):
        img = torch.div(img, max_value)
        return img

    def __len__(self):
        """Function to return the number of records in the dataset"""
        return len(self.data)

    def get_labels(self):
        return (
            np.array(self.data["label"]) > 0
            if self.binary_class
            else self.data["label"]
        )

    def get_label(self, label):
        return label > 0 if self.binary_class else label

    def get_class_weights(self):
        labels = self.get_labels()
        return compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )

    def get_sample_weights(self):
        class_weights = self.get_class_weights()
        labels = self.get_labels()
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        example = self.to_torch_tensor(self.data[index])
        example["image"] = self.scale(example["image"])

        # Base augmentation
        example["image"] = self.base_transform(example["image"])

        # Augmentation
        label = example["label"].item()
        augmentation_prob = self.class_weights[label] / max(self.class_weights)

        if (
            self.data_augm_transform is not None
            and label != 0
            and random.random() < augmentation_prob
        ):
            example["image"] = self.data_augm_transform(example["image"])

        # Normalization
        example["image"] = self.normalization_transform(example["image"])

        # Returns change if test
        if not self.return_all_columns:
            return example["image"], self.get_label(example["label"])
        else:
            example = self.data[index]
            example["label"] = self.get_label(example["label"])
            return example
