import torch

from datasets import load_dataset

from pred_live import (
    load_model,
    create_logits,
    prepare_ecallisto_datasets,
    prepare_dataloaders,
)
from huggingface_hub import hf_hub_download

REPO_ID = "i4ds/flaresense-v2"
MODEL_FILENAME = "model.ckpt"
CONFIG_PATH = "configs/best_v2.yml"


def main(config):
    # Setup WandB API and download the artifact
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

    # Load model and configuration
    model, config = load_model(checkpoint_path, config)

    print("model loaded")

    # Prepare datasets and dataloaders
    ds = load_dataset(
        config["data"]["pred_path"],
        split=config["data"]["pred_split"],
        cache_dir="/mnt/nas05/data01/vincenzo/hu",
    )

    # Create ecallisto dataset and dataloader
    edb = prepare_ecallisto_datasets(ds, config)
    print(edb)

    dataloader = prepare_dataloaders(edb, config["general"]["batch_size"])

    # Predict probabilities
    preds = create_logits(model, dataloader)

    # Save a dataframe with the predictions
    if "manual_label" in ds.column_names:
        ds = ds.select_columns(["manual_label", "start_datetime", "antenna"])
    else:
        ds = ds.select_columns(["start_datetime", "antenna"])
    df = ds.to_pandas()
    df["pred"] = preds
    df.to_csv(
        f"{config['data']['pred_path'].split('/')[1]}_{config['data']['pred_split']}.csv"
    )


if __name__ == "__main__":
    main("configs/best_v2.yml")
