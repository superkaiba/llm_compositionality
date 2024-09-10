#!/bin/bash

for model in EleutherAI/pythia-410m-deduped
do
    export MODEL=$model;
    for dataset in 1 2 3 4
    do
        for epoch in 0 0.125 0.25 1 2 3 4
        do
            export DATASET=$dataset;
            export EPOCH=$epoch;
            sbatch --export=ALL ./surprisal.sh
        done
    done
done
