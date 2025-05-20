#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --job-name flaresense_fit
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=/tmp/vincenzo/huggingface
export TRANSFORMERS_CACHE=/tmp/vincenzo/huggingface/transformers
export HF_DATASETS_CACHE=/tmp/vincenzo/huggingface/datasets

python main.py --config configs/best_v2.yml
