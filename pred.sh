#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name flaresense-pred
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=/tmp/vincenzo/huggingface
export TRANSFORMERS_CACHE=/tmp/vincenzo/huggingface/transformers
export HF_DATASETS_CACHE=/tmp/vincenzo/huggingface/datasets

python pred_dataset.py
