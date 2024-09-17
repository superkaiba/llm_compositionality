#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")

EXP_NAME="final_$current_time"

# Separate into 3 groups so can run in parallel, hope this doesn't cause any bugs with PyTorch cache
EARLY_CHECKPOINT_STEPS=(0 1 2 4 8 16 32 64 128 256 512)
MID_CHECKPOINT_STEPS=(1000 2000 3000 4000 8000 13000 23000 32000 33000 43000)
LATE_CHECKPOINT_STEPS=(53000 63000 64000 73000 83000 93000 103000 113000 123000 133000 143000)
LAST_CHECKPOINT_STEP=(143000)

for ckpt in 128 256 32000 33000 #143000 133000 123000 113000 2 4 32 8 64 8000 83000
do
    for dataset in {1..4}
    do
        export MODEL=6.9b;
        export STEP=$ckpt;
        export DATASET=$dataset;
        sbatch --export=ALL ./run_id.sh
    done
done





