#!/bin/bash
#SBATCH --job-name="id_computation"
#SBATCH -p alien
#SBATCH -t 10-00:00:00
#SBATCH --exclude=node044
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --qos=alien
#SBATCH -o %j_id.out
#SBATCH -e %j_id.err

source  ~/.bashrc;
conda activate adapters;

python3 /home/echeng/llm_compositionality/pca_id.py \
    --dataset $DATASET \
    --step $STEP \
    --model_size $MODEL



