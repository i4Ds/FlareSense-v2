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

The site is a Gradio app (`app.py`). It lists burst images written by `pred_live.py`. It does not run the model itself.

Dev and prod use Docker Compose, the same way as heliotime-v2:

```sh
./du.sh dev deploy
```

The dev site is at <http://localhost:7860/>. It reads `./data/burst_images` unless `FLARESENSE_HOST_IMAGE_DIR` is set.

Production:

```sh
# .env
FLARESENSE_HOST_IMAGE_DIR=/path/to/burst/images
WEB_PORT=7860

./du.sh prod deploy
```

See [deployment](docs/deploy.md). A bare `python app.py` still works: set `FLARESENSE_IMAGE_DIR`, or keep the gitignored `paths.py` the server already uses.

## Evaluation

To reproduce our results, run the following command:

```bash
python main.py --config configs/best_v2.yml
```

## Notebooks

All notebooks can be found in the `_notebooks` directory. They provide
exploratory data analysis, model investigations, and visualizations.

## Current server process

[flaresense.ch](https://flaresense.ch/) still runs as systemd services until it is switched to `./du.sh prod deploy`:

- **flaresense_app.service** — `app.py`
- **flaresense.service** — `pred_live.py` (writes the PNGs the site reads)

```bash
sudo journalctl -u flaresense_app.service
sudo journalctl -u flaresense.service
sudo systemctl restart flaresense_app.service
```

The unit files are on the server at `/etc/systemd/system/`. Image paths for that process live in gitignored `paths.py`, or in `FLARESENSE_IMAGE_DIR` / `FLARESENSE_PREDICT_DIR`.

---

This repository is provided for reference without any warranty.

