#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/slurm/slurm-%j.out

module load miniconda/3
conda activate llm_compositionality

export HF_HOME="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export HF_DATASETS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export TRANSFORMERS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --shuffle