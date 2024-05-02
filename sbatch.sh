#!/bin/sh
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name flaresense-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

wandb agent vincenzo-timmel/FlareSense-v2/1dpxgbtd