#!/bin/sh
#SBATCH --time=56:00:00
#SBATCH --job-name sweep_flaresense_only_spec
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=/tmp/vincenzo/huggingface
export HF_DATASETS_CACHE=/tmp/vincenzo/huggingface/datasets

wandb agent vincenzo-timmel/FlareSense-v2/p98wahrl