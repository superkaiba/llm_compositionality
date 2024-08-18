#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/slurm/slurm-%j.out

module load miniconda/3
conda activate llm_compositionality

# Get the current time in a readable string format
# Get the current time in a format suitable for a directory
current_time=$(date +"%Y%m%d_%H%M%S")
echo "Current time: $current_time"

export HF_HOME="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export HF_DATASETS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export TRANSFORMERS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"

python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 6.9b --shuffle --exp_name final_${current_time}_shuffled
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 12b --exp_name final_${current_time}_ordered
