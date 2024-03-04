import torch
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


# Dataset
class EcallistoData(Dataset):
    def __init__(
        self, dataset, prepro=preprocess, transform=None, return_all_columns=False
    ):
        super().__init__()
        self.data = dataset
        self.prepro = prepro
        self.transform = transform
        self.return_all_columns = return_all_columns

    def __len__(self):
        """Function to return the number of records in the dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset"""
        # Transform images
        if self.prepro:
            img = self.prepro(self.data[index])
        if self.transform:
            img["image"] = self.transform(img["image"])
        if not self.return_all_columns:
            return img["image"], torch.tensor(self.data[index]["label"])
        else:
            return self.data[index]
