# %%
import wandb
from ecallisto_model import ResNet18
import torch

checkpoint_reference = "vincenzo-timmel/FlareSense-v2/model-aen4kqj5:best"

# download checkpoint locally (if not already cached)
run = wandb.init(project="Flaresense-v2-pred", entity="vincenzo-timmel")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

# %%
artifact_dir

# %%
model = ResNet18(2)

# %%
checkpoint = torch.load("artifacts\model-aen4kqj5-v74\model.ckpt", map_location="cpu")

# %%
# load checkpoint
model.load_state_dict(checkpoint["state_dict"])

# %%
from datasets import DatasetDict, load_dataset

ds = load_dataset("i4ds/radio-sunburst-ecallisto")

# %%
dd = DatasetDict()
dd["train"] = ds["train"]
dd["test"] = ds["test"]
dd["validation"] = ds["validation"]

# %%
# Define normalization
import yaml
from ecallisto_model import (
    ResNet18,
    create_normalize_function,
    create_unnormalize_function,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose, Resize
from ecallisto_dataset import (
    EcallistoDatasetBinary,
)

base_transform = Compose(
    [
        Resize([224, 224]),  # Resize the image
    ]
)

with open("antenna_stats.yaml", "r") as file:
    antenna_stats = yaml.safe_load(file)
normalize_transform = create_normalize_function(
    antenna_stats=antenna_stats, simple=False
)

# %%
ds_train = EcallistoDatasetBinary(
    dd["train"],
    base_transform=base_transform,
    normalization_transform=normalize_transform,
)
ds_valid = EcallistoDatasetBinary(
    dd["validation"],
    base_transform=base_transform,
    normalization_transform=normalize_transform,
)
ds_test = EcallistoDatasetBinary(
    dd["test"],
    base_transform=base_transform,
    normalization_transform=normalize_transform,
)


# %%
train_dataloader = DataLoader(
    ds_train,
    batch_size=32,
    num_workers=0,
    shuffle=False,
    persistent_workers=True,
)

val_dataloader = DataLoader(
    ds_valid,
    batch_size=32,
    num_workers=0,
    shuffle=False,
    persistent_workers=True,
)

test_dataloader = DataLoader(
    ds_test,
    batch_size=32,
    num_workers=0,
    shuffle=True,  # To randomly log images
    persistent_workers=False,
)

# %%
model.eval()  # Ensure the model is in evaluation mode

data = []  # To store all the information

with torch.no_grad():
    for inputs, labels, antennas, datetimes in test_dataloader:
        logits = model(inputs)

        for idx in range(inputs.shape[0]):
            data.append(
                {
                    "probability_class_0": logits[idx][0].item(),
                    "probability_class_1": logits[idx][1].item(),
                    "antenna": antennas[idx],
                    "datetime": datetimes[idx],
                    "true_label": labels[idx].item(),
                    # Optional: Add predicted class if useful
                    "predicted_class": logits[idx].argmax().item(),
                }
            )

# %%
# Convert collected data into a DataFrame
df = pd.DataFrame(data)
