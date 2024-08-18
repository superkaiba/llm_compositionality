#!/bin/bash
#SBATCH --job-name="opt_id_extraction"
#SBATCH -p alien
#SBATCH -t 10-00:00:00
#SBATCH --exclude=node044
#SBATCH --mem=380G
#SBATCH --cpus-per-task=4
#SBATCH --qos=alien
#SBATCH -o opt_id.out
#SBATCH -e opt_id.err

source  ~/.bashrc;
conda activate adapters;


python3 /home/echeng/llm_compositionality/pca_id.py \
    --dataset $DATASET \
    --epoch $EPOCH \



