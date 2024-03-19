#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name flaresense-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err

python3 create_yaml_mean_std_min_max_antenna.py
# srun python3 main.py --config configs/t100.yml