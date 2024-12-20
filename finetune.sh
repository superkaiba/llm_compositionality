#!/bin/bash
#SBATCH --job-name="finetune"
#SBATCH -p alien
#SBATCH -t 10-00:00:00
#SBATCH --exclude=node044
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH -o %j_finetune.o
#SBATCH -e %j_finetune.e

source  ~/.bashrc;
conda activate control;
cd /home/echeng/llm_compositionality;

python3 /home/echeng/llm_compositionality/finetune.py \
    --model_name $MODEL \
    --dataset $DATASET



