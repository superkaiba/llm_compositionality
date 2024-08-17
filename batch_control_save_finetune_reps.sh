#!/bin/bash

for epoch in 0 #1 2 3 #EleutherAI/pythia-410m-deduped
do
    export EPOCH=$epoch;
    for dataset in 1 2 3 4
    do
        export DATASET=$dataset;
        sbatch --export=ALL ./save_finetune_reps.sh
    done
done
