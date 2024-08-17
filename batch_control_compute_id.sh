#!/bin/bash

for epoch in 0 1 2 3 4
do
    for dataset in 1 2 3 4
    do
        export DATASET=$dataset;
        export EPOCH=$epoch;
        sbatch --export=ALL ./compute_id.sh
    done
done
