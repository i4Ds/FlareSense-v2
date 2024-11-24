#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name flaresense-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=/tmp/vincenzo/huggingface
srun python pred.py