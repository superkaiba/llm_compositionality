#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")

# MODEL_SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
MODEL_SIZES=("12b")
SHUFFLES=("True" "False")

for model_size in ${MODEL_SIZES[@]}; do
    # for shuffle in ${SHUFFLES[@]}; do
    #     if [ "$shuffle" = "True" ]; then
    #         shuffle_str="shuffled"
    #     else
    #         shuffle_str="ordered"
    #     fi
        EXP_NAME="final_$current_time"
        echo "Running $model_size"
        if [ "$model_size" = "12b" ]; then
            sbatch run_big.sh $model_size $EXP_NAME
        else
            sbatch run.sh $model_size $EXP_NAME
        fi
    # done
done
