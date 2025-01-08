import torch

from datasets import load_dataset


import wandb

from pred_live import (
    create_logits,
    load_model,
    prepare_ecallisto_datasets,
    prepare_dataloaders,
)


def main(checkpoint_reference, config):
    # Setup WandB API and download the artifact
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    _ = artifact.download()

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and configuration
    print(artifact.file())
    model, config = load_model(artifact.file(), config)

    print("model loaded")

    # Prepare datasets and dataloaders
    ds = load_dataset(config["data"]["pred_path"], split=config["data"]["pred_split"])

    # Create ecallisto dataset and dataloader
    edb = prepare_ecallisto_datasets(ds, config)
    print(edb)

    dataloader = prepare_dataloaders(edb, config["general"]["batch_size"])

    # Predict probabilities
    preds = create_logits(model, dataloader, device)

    # Save a dataframe with the predictions
    ds = ds.select_columns(["datetime", "antenna"])
    df = ds.to_pandas()
    df["pred"] = preds
    df.to_csv(f"{config['data']['pred_path'].split('/')[1]}_test.csv")


if __name__ == "__main__":
    main("vincenzo-timmel/FlareSense-v2/final_model:v7", "configs/best_v2.yml")
