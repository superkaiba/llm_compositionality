#!/bin/bash

for model in EleutherAI/pythia-410m-deduped
do
    export MODEL=$model;
    for dataset in 1 2 3 4
    do
        export DATASET=$dataset;
        sbatch --export=ALL ./finetune.sh
    done
done
