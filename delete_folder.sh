#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --job-name clean_tmp-v2
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --partition=p4500
#SBATCH --out=logs/%j.out
#SBATCH --error=logs/%j.err


rm -rf /tmp/vincenzo/