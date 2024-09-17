#!/bin/bash
#SBATCH --job-name="id_computation"
#SBATCH --gres=gpu:1
#SBATCH -p alien
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --exclude=node044
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --qos=alien
#SBATCH -o /home/echeng/llm_compositionality/logs/%j.out
#SBATCH -e /home/echeng/llm_compositionality/logs/%j.err

source  ~/.bashrc;
conda activate control;

# Get the current time in a readable string format
# Get the current time in a format suitable for a directory
current_time=$(date +"%Y%m%d_%H%M%S")
echo "Current time: $current_time"

export model_size="$1"
export exp_name="$2"
export checkpoint_steps=($3)
export n_words_correlated=(1 2 3 4)
export batch_size=64
export device=cuda
export data_dir="/home/echeng/llm_compositionality/data" # TODO: EMILY CHANGE THIS
export results_path="/home/echeng/llm_compositionality/results_new" # TODO: EMILY CHANGE THIS

export HF_HOME="/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS
export HF_DATASETS_CACHE="/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS
export TRANSFORMERS_CACHE="/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS

# Running shuffle and non shuffle in same script because or else the PyTorch cache bugs because they try to load the model at the same time

for checkpoint_step in ${checkpoint_steps[*]}
do
    python3 /home/echeng/llm_compositionality/run_pipeline.py \
        --model_size $model_size \
        --exp_name ${exp_name}_ordered \
        --checkpoint_step $checkpoint_step \
        --n_words_correlated ${n_words_correlated[*]} \
        --batch_size $batch_size \
        --device $device \
        --data_dir $data_dir \
        --results_path $results_path
done

#python3 /home/echeng/llm_compositionality/run_pipeline.py \
#    --model_size $model_size \
    # --shuffle \
    # --exp_name ${exp_name}_shuffled \
    # --checkpoint_steps ${checkpoint_steps[*]} \
    # --n_words_correlated ${n_words_correlated[*]} \
    # --batch_size $batch_size \
    # --device $device \
    # --data_dir $data_dir \
    # --results_path $results_path

