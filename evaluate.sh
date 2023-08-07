#!/bin/bash
#SBATCH --job-name=MusicTransformer
#SBATCH --mem=30G
#SBATCH -c1
#SBATCH --gres=gpu:1,gpumem:24G
#SBATCH -t 1-00:00
module load cuda/11.4

python3 -u evaluate.py -model_weights rpr/results/best_acc_weights.pickle --rpr
