#!/bin/sh
#SBATCH --time=20:00:00
#SBATCH --job-name sweep_bt_flaresense
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=/tmp/vincenzo/huggingface
python main.py --config configs/test_v2.yml