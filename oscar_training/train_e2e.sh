#!/bin/bash

#SBATCH -p gpu --gres=gpu:1

#SBATCH -n 4
#SBATCH --mem-per-cpu=10G
#SBATCH -t 04:00:00

#SBATCH -o ./oscar_job_scripts/output/train_e2e_400p_all_10epoch_bs_96.out

python main.py --end-to-end