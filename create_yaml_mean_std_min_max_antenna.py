import torch
from collections import defaultdict
import yaml
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor

# Initialize dictionary to hold the statistics for each antenna
antenna_stats = defaultdict(
    lambda: {
        "mean": 0,
        "M2": 0,
        "n_elements": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }
)

# Create dataset
ds = load_dataset("i4ds/radio-sunburst-ecallisto")


for entry in tqdm(ds["train"], desc="Processing dataset"):
    x = pil_to_tensor(entry["image"]).float()
    antenna = entry["antenna"]

    # Update count, mean, and M2
    n = x.numel()
    new_mean = torch.mean(x).item()
    new_variance = torch.var(
        x, unbiased=False
    ).item()  # Use unbiased=False for a population variance
    new_count = n

    # Retrieve current stats for the antenna
    cur_stats = antenna_stats[antenna]
    delta = new_mean - cur_stats["mean"]
    cur_stats["mean"] += delta * new_count / (cur_stats["n_elements"] + new_count)
    cur_stats["M2"] += new_variance * new_count + delta**2 * cur_stats[
        "n_elements"
    ] * new_count / (cur_stats["n_elements"] + new_count)
    cur_stats["n_elements"] += new_count

    # Update min and max
    cur_stats["min"] = min(cur_stats["min"], torch.min(x).item())
    cur_stats["max"] = max(cur_stats["max"], torch.max(x).item())

    # Update the dictionary
    antenna_stats[antenna] = cur_stats

# Finalize variance and std calculation and store results
final_stats = {}
for antenna, stats in antenna_stats.items():
    variance = stats["M2"] / stats["n_elements"]  # This is the population variance
    std = torch.sqrt(torch.tensor(variance)).item()
    final_stats[antenna] = {
        "mean": stats["mean"],
        "std": std,
        "min": stats["min"],
        "max": stats["max"],
    }

# Save to YAML
with open("antenna_stats.yaml", "w") as file:
    yaml.dump(final_stats, file, default_flow_style=False)
