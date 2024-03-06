#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name flaresense-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

srun python3 main.py --config configs/t200.yml