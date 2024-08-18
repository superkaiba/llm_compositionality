#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/slurm/slurm-%j.out

module load miniconda/3
conda activate llm_compositionality

# Get the current time in a readable string format
# Get the current time in a format suitable for a directory
current_time=$(date +"%Y%m%d_%H%M%S")
echo "Current time: $current_time"

export model_size=$1
export exp_name=$2
export shuffle=$3

export HF_HOME="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export HF_DATASETS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export TRANSFORMERS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"


if [ "$shuffle" = "True" ]; then
    python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --shuffle --exp_name $exp_name
else
    python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --exp_name $exp_name
fi