#!/bin/sh
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name flaresense-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=logs/stt_eval_%j.out
#SBATCH --error=logs/stt_eval_%j.err

python3 main.py