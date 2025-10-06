# FlareSense v2

FlareSense is an experimental project that trains a convolutional neural network
on e-Callisto radio spectrograms to detect solar radio bursts. The repository
contains training code, prediction utilities, a web app for user interaction,
and several notebooks used during development.

## Installation

The code requires **Python 3.11**. Use the conda environment for dependencies:

```bash
conda activate flaresense-v2
pip install -r requirements.txt
```

## Training

Training is driven by YAML configuration files. A typical run looks like:

```bash
python main.py --config configs/best_v2.yml
```

The `main.sh` script shows how to submit a job on a SLURM cluster.

## Prediction

Run inference on a dataset with:

```bash
python pred_dataset.py
```

For live prediction through a Gradio interface execute:

```bash
python pred_live.py
```

## Web App

The web app in `app.py` provides a user interface for uploading data, running predictions, and viewing results. It is built with Flask and integrates with the prediction models.

## Evaluation

To reproduce our results, run the following command:

```bash
python main.py --config configs/best_v2.yml
```

## Notebooks

All notebooks can be found in the `_notebooks` directory. They provide
exploratory data analysis, model investigations, and visualizations.

## Deployment

FlareSense is deployed as Linux systemd services for production use:

- **flaresense_app.service**: Manages the web app (`app.py`) for user-facing interactions.
- **flaresense.service**: Handles continuous prediction tasks.

## Service Management

If FlareSense is deployed as systemd services, you can inspect the logs with:

```bash
sudo journalctl -u flaresense_app.service  # For web app logs
sudo journalctl -u flaresense.service      # For prediction service logs
```

After modifying the service files, redeploy with:

```bash
sudo systemctl restart flaresense_app.service
sudo systemctl restart flaresense.service
sudo systemctl daemon-reload
```

The .serice-file can be found here: /etc/systemd/system/flaresense_app.service
---

This repository is provided for reference without any warranty.

