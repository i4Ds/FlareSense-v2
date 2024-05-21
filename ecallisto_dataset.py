import random

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Resize
import torch
from torchvision import transforms
import os
import pandas as pd
import torch
import torch.nn.functional as F


# Dataset
class EcallistoDataset(Dataset):
    def __init__(
        self, dataset, resize_func, normalization_transform, data_augm_transform=None
    ):
        super().__init__()
        self.data = dataset
        self.data_augm_transform = data_augm_transform
        self.resize_func = resize_func
        self.normalization_transform = normalization_transform
        self.dataset_label_weight = self.get_dataset_label_weight()

    @staticmethod
    def to_torch_tensor(example):
        # Convert the example to a torch tensor
        if "file_path" in example:
            # It's a parquet file, containing a DF
            df = pd.read_parquet(example["file_path"])
            example["image"] = torch.from_numpy(df.values.T).float()
        else:
            example["image"] = pil_to_tensor(example["image"]).float()
        example["label"] = torch.tensor(example["label"])
        return example

    def __len__(self):
        """Function to return the number of records in the dataset"""
        return len(self.data)

    def get_labels(self):
        return self.get_dataset_labels()

    def get_dataset_label_weight(self):
        labels = self.get_dataset_labels()
        return compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )

    def get_dataset_labels(self):
        return self.data["label"]

    def get_class_weights(self):
        return self.dataset_label_weight

    def get_sample_weights(self):
        class_weights = self.dataset_label_weight
        labels = self.get_labels()
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        example = self.to_torch_tensor(self.data[index])

        # Normalization
        example["image"] = self.normalization_transform(example["image"])
        # Resize
        example["image"] = self.resize_func(example["image"])
        # Data aug
        if self.data_augm_transform is not None:
            example["image"] = self.data_augm_transform(example["image"])

        # Returns all
        return (
            example["image"].unsqueeze(0),
            example["label"],
            example["antenna"],
            example["datetime"],
        )

    def save_image(self, image_tensor, antenna, datetime):
        # Create a directory to save the image
        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)

        # Convert datetime to a filename-friendly format
        datetime_str = datetime.replace(":", "-").replace(" ", "_")

        # Define the filename
        filename = f"{antenna}_{datetime_str}.png"
        file_path = os.path.join(save_dir, filename)

        # Convert the tensor to a PIL Image and save it
        image = transforms.ToPILImage()(image_tensor)
        image.save(file_path)
        print(f"Image saved to {file_path}")


class EcallistoDatasetBinary(EcallistoDataset):
    def __init__(
        self,
        dataset,
        resize_func,
        normalization_transform,
        data_augm_transform=None,
    ):
        # Initialize the parent class
        super().__init__(
            dataset,
            resize_func,
            normalization_transform,
            data_augm_transform=data_augm_transform,
        )

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        image, label, antenna, datetime = super().__getitem__(index)

        # Convert label to binary
        label = 0 if label.item() == 0 else 1

        return image, torch.tensor(label), antenna, datetime

    def get_labels(self):
        # Return binary labels: 0 if the label is 0, 1 otherwise
        return np.where(np.array(self.data["label"]) == 0, 0, 1)

    def get_class_weights(self):
        labels = self.get_labels()
        labels = np.where(labels != 0, 1, 0)
        cw = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return cw


class CustomSpecAugment:
    def __init__(self, frequency_masking_para, method="max"):
        self.frequency_masking_para = frequency_masking_para
        self.method = method

    def __call__(self, spectrogram):
        # Calculate per-row padding values based on the specified method
        if self.method == "max":
            padding_values = torch.max(spectrogram, dim=1).values
        elif self.method == "min":
            padding_values = torch.min(spectrogram, dim=1).values
        elif self.method == "median":
            padding_values = torch.median(spectrogram, dim=1).values
        elif self.method == "mean":
            padding_values = torch.mean(spectrogram, dim=1).values
        else:
            raise ValueError(
                "Invalid method parameter. Choose from 'max', 'min', 'median', or 'mean'."
            )

        # Apply frequency masking
        spectrogram = self.frequency_mask(spectrogram, padding_values)
        return spectrogram

    def frequency_mask(self, spectrogram, padding_values):
        num_mel_channels = spectrogram.size(0)
        mask_param = torch.randint(0, self.frequency_masking_para, (1,)).item()

        f = torch.randint(0, num_mel_channels - mask_param, (1,)).item()
        mask_value = padding_values[f]
        spectrogram[f : f + mask_param, :] = mask_value
        return spectrogram


def custom_resize(spectrogram, target_size):
    spectrogram = custom_resize_height(spectrogram, target_size)
    spectrogram = custom_resize_width_max(spectrogram, target_size)
    return spectrogram


def custom_resize_height(spectrogram, target_size):
    resize = Resize((target_size[0], spectrogram.shape[1]))
    return resize(spectrogram.unsqueeze(0)).squeeze(0)


def custom_resize_width_max(spectrogram, target_size):
    """
    Resize the spectrogram by aggregating using the max value within each window.

    Args:
        spectrogram (torch.Tensor): Input spectrogram of shape (H, W).
        target_size (tuple): Desired output size (target_height, target_width).

    Returns:
        torch.Tensor: Resized spectrogram.
    """
    H, _ = spectrogram.shape

    spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension if missing

    # Apply adaptive max pooling
    pooled_spectrogram = F.adaptive_max_pool2d(
        spectrogram, output_size=(H, target_size[1])
    )

    return pooled_spectrogram.squeeze(0)  # Remove the channel dimension if it was added


def remove_background(spectrogram):
    # Calculate the median of each row
    median_values = torch.median(spectrogram, dim=1).values

    # Subtract the median from each row
    background_removed = spectrogram - median_values[:, None]

    return background_removed


def global_min_max_scale(spectrogram):
    # Calculate the global minimum and maximum
    min_value = torch.min(spectrogram)
    max_value = torch.max(spectrogram)

    # Apply global Min-Max scaling
    scaled_spectrogram = (spectrogram - min_value) / (max_value - min_value)

    return scaled_spectrogram


def preprocess_spectrogram(spectrogram):
    # Remove background
    spectrogram = remove_background(spectrogram)

    # Apply global Min-Max scaling
    spectrogram = global_min_max_scale(spectrogram)

    return spectrogram


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


def filter_antennas(dataset, antenna_list):
    """
    Filter out samples from a dataset based on a list of antenna names.

    :param dataset: The dataset to process (assumed to be a Hugging Face dataset).
    :param antenna_list: List of antenna names to keep.
    :return: A new dataset with only the samples from the specified antennas.
    """

    # Get all indices for the specified antennas
    selected_indices = [
        i for i, antenna in enumerate(dataset["antenna"]) if antenna in antenna_list
    ]

    # Create and return the filtered dataset
    filtered_dataset = dataset.select(selected_indices)

    return filtered_dataset
