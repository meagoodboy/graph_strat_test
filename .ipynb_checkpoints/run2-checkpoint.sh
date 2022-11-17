#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=./run2_log.txt
#SBATCH --mail-type=END

python3 graph_stratification.py ./config/config_sm_2.yaml