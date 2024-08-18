#!/bin/bash
#SBATCH --job-name="surprisal"
#SBATCH -p alien
#SBATCH -t 10-00:00:00
#SBATCH --exclude=node044
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH -o %j_bootstrap.o
#SBATCH -e %j_bootstrap.e

source  ~/.bashrc;
conda activate control;
cd /home/echeng/llm_compositionality;

python3 /home/echeng/llm_compositionality/compute_surprisal.py \
    --model_name $MODEL \
    --dataset $DATASET \
    --epoch $EPOCH



