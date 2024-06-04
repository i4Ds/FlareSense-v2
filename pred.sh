#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name flaresense-pred
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

wandb agent vincenzo-timmel/FlareSense-v2/tnkcy9ek