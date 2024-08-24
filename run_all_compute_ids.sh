#!/bin/bash

for base_path in /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_ordered_20240815_013151 /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_shuffled_20240815_013501
do 
for model_size in 6.9b
do 
    sbatch /home/mila/t/thomas.jiralerspong/llm_compositionality/run_compute_ids.sh $base_path $model_size 
done
done