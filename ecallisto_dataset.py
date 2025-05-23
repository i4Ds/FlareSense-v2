import atexit
import os
import shutil
import signal
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image


# Dataset
class EcallistoDataset(Dataset):
    def __init__(
        self,
        dataset=None,
        label_name="label",
        resize_func=None,
        normalization_transform=None,
        cache=False,
        delete_cache_after_run=False,
        cache_base_dir="/tmp/vincenzo/CAN_BE_DELETED/ecallisto",
        augm_before_resize=None,
        augm_after_resize=None,
    ):
        if dataset is None:
            print(
                f"Warning: dataset is None. This is OK if you are using this class for inference."
            )
        super().__init__()
        self.data = dataset
        self.label_name = label_name
        self.normalization_transform = normalization_transform
        self.augm_before_resize = augm_before_resize
        self.augm_after_resize = augm_after_resize
        self.resize_func = resize_func
        # Cleanup parameters.
        self.cache = cache
        self.cache_dir = os.path.join(cache_base_dir, str(uuid4()))

        # Process, if dataset is availble
        if self.data is not None:
            self.dataset_label_weight = self.get_dataset_label_weight()

            if delete_cache_after_run:
                atexit.register(self.clean_up)
                signal.signal(signal.SIGTERM, self.clean_up)
                signal.signal(signal.SIGINT, self.clean_up)

        # Setup some torch functions
        self.to_tensor = ToTensor()

    def image_to_torch_tensor(self, example):
        # Convert the example to a torch tensor
        if "file_path" in example:
            # It's a parquet file, containing a DF
            df = pd.read_parquet(example["file_path"])
            image = torch.from_numpy(df.values.T).float()
        elif isinstance(example["image"], Image.Image):
            # Handle the case where example["image"] is already a PIL Image
            image = self.to_tensor(example["image"])
        elif "bytes" in example["image"]:
            image = Image.open(BytesIO(example["image"]["bytes"]))
            image = self.to_tensor(image)
        else:
            image = pil_to_tensor(example["image"]).float()
        # Check if batched dim 0 exists
        if len(image.shape) == 3:
            image = image.squeeze(0)
        return image

    def __len__(self):
        """Function to return the number of records in the dataset"""
        return len(self.data)

    def clean_up(self, *args, **kwargs):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def __del__(self):
        self.clean_up()

    def get_labels(self):
        return self.get_dataset_labels()

    def get_dataset_label_weight(self):
        labels = self.get_dataset_labels()
        return compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )

    def get_dataset_labels(self):
        return self.data[self.label_name]

    def get_class_weights(self):
        return self.dataset_label_weight

    def get_sample_weights(self):
        class_weights = self.dataset_label_weight
        labels = self.get_labels()
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights

    def create_cache_path(self, example):
        return Path(
            self.cache_dir,
            example["antenna"],
            example["datetime"] + ".torch",
        )

    def augment_image(self, image):
        # Data augmentation before resize
        if self.augm_before_resize is not None:
            image = self.augm_before_resize(image)

        # Resize
        if self.resize_func is not None:
            image = self.resize_func(image)

        # Data augmentation after resize
        if self.augm_after_resize is not None:
            image = self.augm_after_resize(image)

        return image

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        example = self.data[index]

        # Make some fixes about columns
        if not "datetime" in example:
            example["datetime"] = example["start_datetime"]
            del example["start_datetime"]

        # Type casting
        if not isinstance(example["datetime"], str):
            example["datetime"] = str(example["datetime"])

        # Caching
        example_image_path = self.create_cache_path(example)
        if not self.cache or not os.path.exists(example_image_path):
            image = self.image_to_torch_tensor(example)
            image = self.normalization_transform(image)
            if self.cache:
                os.makedirs(os.path.dirname(example_image_path), exist_ok=True)
                try:
                    torch.save(image, example_image_path)
                except Exception as e:
                    print(f"Error saving image to cache: {e}")
                    print(f"Image path: {example_image_path}")
        else:
            image = torch.load(example_image_path)

        # Augmentation
        image = self.augment_image(image)

        # Min max scale image
        image = min_max_scaler(image, (0, 1))

        # Create example
        example[self.label_name] = torch.tensor(example[self.label_name])
        example["image"] = image

        return (
            example["image"].unsqueeze(0),
            example[self.label_name].unsqueeze(0),
            example["antenna"],
            example["datetime"],
        )


class EcallistoDatasetLive(Dataset):
    def __init__(
        self,
        dataset,
        normalization_transform,
    ):
        self.dataset = dataset

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        image = self.dataset[index]
        tensor = pil_to_tensor(image).float()
        tensor = tensor.squeeze(0)
        tensor = min_max_scaler(tensor)

        label = label.squeeze(0)

        return image


class EcallistoDatasetBinary(EcallistoDataset):
    def __init__(
        self,
        dataset=None,
        label_name="label",
        resize_func=None,
        normalization_transform=None,
        augm_before_resize=None,
        augm_after_resize=None,
        cache=False,
    ):
        # Initialize the parent class
        super().__init__(
            dataset,
            label_name,
            resize_func,
            normalization_transform,
            augm_before_resize=augm_before_resize,
            augm_after_resize=augm_after_resize,
            cache=cache,
        )

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        image, label, antenna, datetime = super().__getitem__(index)

        label = label.squeeze(0)

        # Convert label to binary
        label = 0 if label.detach() == 0 else 1

        # Save the image tensor as PNG if the index is 0
        if index == 0:
            pil_image = to_pil_image(image)  # Convert tensor to PIL image
            pil_image.save("image_tensor.png")  # Save the image as PNG
            print("Image tensor saved as PNG for index 0")

        return image, torch.tensor(label).float().unsqueeze(0), antenna, datetime

    def get_labels(self):
        # Return binary labels: 0 if the label is 0, 1 otherwise
        return np.where(np.array(self.data[self.label_name]) == 0, 0, 1)

    def get_class_weights(self):
        labels = self.get_labels()
        cw = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        print(f"{cw.shape=}. {cw=}")
        return cw[1]  # Binary


class CustomSpecAugment:
    def __init__(self, frequency_masking_para, time_masking_para, method="max"):
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.method = method

    def __call__(self, spectrogram):
        if self.frequency_masking_para == 0 and self.time_masking_para == 0:
            return spectrogram

        # Apply frequency masking if parameter is set
        if self.frequency_masking_para > 0:
            padding_values_freq = self.get_padding_value(spectrogram, self.method)
            spectrogram = self.frequency_mask(spectrogram, padding_values_freq)

        # Apply time masking if parameter is set
        if self.time_masking_para > 0:
            padding_values_time = self.get_padding_value(spectrogram, self.method)
            spectrogram = self.time_mask(spectrogram, padding_values_time)

        return spectrogram

    def get_padding_value(self, spectrogram, method):
        if method == "max":
            padding_value = torch.max(spectrogram).detach()
        elif method == "min":
            padding_value = torch.min(spectrogram).detach()
        elif method == "median":
            padding_value = torch.median(spectrogram).detach()
        elif method == "mean":
            padding_value = torch.mean(spectrogram).detach()
        elif method == "random":
            padding_value = torch.rand(1).detach()
        else:
            raise ValueError(
                "Invalid method parameter. Choose from 'max', 'min', 'median', 'mean', or 'random'."
            )
        return padding_value

    def frequency_mask(self, spectrogram, padding_values):
        num_mel_channels = spectrogram.size(0)
        mask_param = torch.randint(0, self.frequency_masking_para + 1, (1,)).detach()
        f = torch.randint(0, num_mel_channels - mask_param + 1, (1,)).detach()
        spectrogram[f : f + mask_param, :] = padding_values
        return spectrogram

    def time_mask(self, spectrogram, padding_values):
        time_steps = spectrogram.size(1)
        mask_param = torch.randint(0, self.time_masking_para + 1, (1,)).detach()
        t = torch.randint(0, time_steps - mask_param + 1, (1,)).detach()
        spectrogram[:, t : t + mask_param] = padding_values
        return spectrogram


class TimeWarpAugmenter:
    def __init__(self, W=50):
        """
        Initialize the TimeWarpAugmenter with the strength of warp (W).
        """
        self.W = W

    def __call__(self, specs):
        """
        Apply time warp augmentation when the class is called.

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        """
        if self.W == 0:
            return specs
        if specs.size(-1) <= 2 * self.W:
            self.W = specs.size(-1) // 2  # Adjust W if it's too large for the input
            print(
                f"Spec is too short to do time warping. Got: {specs.size(-1)}. Expected (min.): {2 * self.W + 1}\nAjusting W to {self.W}"
            )
            return specs
        if not torch.is_tensor(specs):
            specs = torch.from_numpy(specs)
        if specs.dim() < 2 or specs.dim() > 3:
            raise ValueError("You sure it's a Spectrogram?")
        if specs.dim() == 2:
            # Add dummy batch.
            specs = torch.unsqueeze(specs, dim=0)
        warped = self.time_warp(specs, self.W)
        return warped.squeeze(0)

    @staticmethod
    def h_poly(t):
        tt = t.unsqueeze(-2) ** torch.arange(4, device=t.device).view(-1, 1)
        A = torch.tensor(
            [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
            dtype=t.dtype,
            device=t.device,
        )
        return A @ tt

    @staticmethod
    def hspline_interpolate_1D(x, y, xs):
        """
        Input x and y must be of shape (batch, n) or (n)
        """
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        # print(torch.abs(x.take_along_dim(idxs+1, dim=-1) - x.gather(dim=-1, index=idxs+1)))
        dx = x.gather(dim=-1, index=idxs + 1) - x.gather(dim=-1, index=idxs)
        hh = TimeWarpAugmenter.h_poly((xs - x.gather(dim=-1, index=idxs)) / dx)
        return (
            hh[..., 0, :] * y.gather(dim=-1, index=idxs)
            + hh[..., 1, :] * m.gather(dim=-1, index=idxs) * dx
            + hh[..., 2, :] * y.gather(dim=-1, index=idxs + 1)
            + hh[..., 3, :] * m.gather(dim=-1, index=idxs + 1) * dx
        )

    def time_warp(self, specs, W=80):
        """
        Timewarp augmentation by https://github.com/IMLHF/SpecAugmentPyTorch/blob/master/spec_augment_pytorch.py

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        W: strength of warp
        """
        device = specs.device
        specs = specs.unsqueeze(0)  # Add dim for channels
        batch_size, _, num_rows, spec_len = specs.shape

        warp_p = torch.randint(W, spec_len - W, (batch_size,), device=device)

        # Uniform distribution from (0,W) with chance to be up to W negative
        # warp_d = torch.randn(1)*W # Not using this since the paper author make random number with uniform distribution
        warp_d = torch.randint(-W, W, (batch_size,), device=device)
        # print("warp_d", warp_d)
        x = torch.stack(
            [
                torch.tensor([0], device=device).expand(batch_size),
                warp_p,
                torch.tensor([spec_len - 1], device=device).expand(batch_size),
            ],
            1,
        )
        y = torch.stack(
            [
                torch.tensor([-1.0], device=device).expand(batch_size),
                (warp_p - warp_d) * 2 / (spec_len - 1.0) - 1.0,
                torch.tensor([1.0], device=device).expand(batch_size),
            ],
            1,
        )

        # Interpolate from 3 points to spec_len
        xs = (
            torch.linspace(0, spec_len - 1, spec_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        ys = TimeWarpAugmenter.hspline_interpolate_1D(x, y, xs)

        grid = torch.cat(
            (
                ys.view(batch_size, 1, -1, 1).expand(-1, num_rows, -1, -1),
                torch.linspace(-1, 1, num_rows, device=device)
                .view(-1, 1, 1)
                .expand(batch_size, -1, spec_len, -1),
            ),
            -1,
        )

        return torch.nn.functional.grid_sample(specs, grid, align_corners=True).squeeze(
            0
        )


def custom_resize(spectrogram, target_size):
    spectrogram = custom_resize_height(spectrogram, target_size)
    spectrogram = custom_resize_width_max(spectrogram, target_size)
    return spectrogram


def normal_resize(spectrogram, target_size):
    resize = Resize(target_size)
    return resize(spectrogram.unsqueeze(0)).squeeze(0)


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


def min_max_scaler(tensor, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor to scale.
        feature_range (tuple): Desired range of transformed data (default is (0, 1)).

    Returns:
        torch.Tensor: Scaled tensor.
    """
    min_val, max_val = feature_range
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    scaled_tensor = scaled_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def clip_tensor(tensor, min_val, max_val):
    """
    Clip (limit) the values in a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        min_val (float): Minimum value to clip to.
        max_val (float): Maximum value to clip to.

    Returns:
        torch.Tensor: Clipped tensor.
    """
    return torch.clamp(tensor, min=min_val, max=max_val)


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
