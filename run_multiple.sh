#!/bin/bash
EXP_NAME="final_results"
MODEL_SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
SHUFFLES=("True" "False")

for model_size in ${MODEL_SIZES[@]}; do
    for shuffle in ${SHUFFLES[@]}; do
        echo "Running $model_size with shuffle $shuffle"
        if [ "$shuffle" == "True" ]; then
            sbatch run.sh $model_size True $EXP_NAME
        else
            sbatch run.sh $model_size False $EXP_NAME
        fi
    done
done
