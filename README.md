# FlareSense v2

FlareSense is an experimental project that trains a convolutional neural network
on e-Callisto radio spectrograms to detect solar radio bursts. The repository
contains training code, prediction utilities and several notebooks used during
development.

## Installation

The code requires **Python 3.11**. Install the dependencies with:

```bash
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

## Notebooks

All notebooks can be found in the `_notebooks` directory. They provide
exploratory data analysis and model investigations.

## Service management

If FlareSense is deployed as a systemd service you can inspect the logs with:

```bash
sudo journalctl -u flaresense.service
```

After modifying the service files redeploy with:

```bash
sudo systemctl restart flaresense.service
sudo systemctl daemon-reload
```

---

This repository is provided for reference without any warranty.

