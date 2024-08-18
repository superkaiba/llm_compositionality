#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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
# export shuffle=$3

export HF_HOME="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export HF_DATASETS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
export TRANSFORMERS_CACHE="/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"

python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --exp_name ${exp_name}_ordered
python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --shuffle --exp_name ${exp_name}_shuffled

# if [ "$shuffle" = "True" ]; then
#     python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --shuffle --exp_name $exp_name
# else
#     python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size $model_size --exp_name $exp_name
# fi
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 160m --exp_name final_ordered_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 410m --exp_name final_ordered_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 1b --exp_name final_ordered_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 1.4b --exp_name final_ordered_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 2.8b --exp_name final_ordered_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 6.9b --exp_name final_ordered_$current_time   
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 12b --exp_name final_ordered_$current_time    

# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 70m --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 160m --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 410m --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 1b --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 1.4b --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 2.8b --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 6.9b --shuffle --exp_name final_shuffled_$current_time
# python /home/mila/t/thomas.jiralerspong/llm_compositionality/run_pipeline.py --model_size 12b --shuffle --exp_name final_shuffled_$current_time
