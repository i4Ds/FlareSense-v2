#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --job-name sweep_bt_flaresense
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

wandb agent vincenzo-timmel/FlareSense-Barlow-Twins/d53x3hjd