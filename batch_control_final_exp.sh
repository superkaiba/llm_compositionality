#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")

EXP_NAME="final_$current_time"

# Separate into 3 groups so can run in parallel, hope this doesn't cause any bugs with PyTorch cache
EARLY_CHECKPOINT_STEPS=(0 1 2 4 8 16 32 64 128 256 512)
MID_CHECKPOINT_STEPS=(1000 2000 3000 4000 8000 13000 23000 32000 33000 43000)
LATE_CHECKPOINT_STEPS=(53000 63000 64000 73000 83000 93000 103000 113000 123000 133000 143000)
LAST_CHECKPOINT_STEP=(143000)

for checkpoint_step in ${EARLY_CHECKPOINT_STEPS[*]}
do
    sbatch run.sh 70m $EXP_NAME $checkpoint_step
done

for checkpoint_step in ${MID_CHECKPOINT_STEPS[*]}
do
    sbatch run.sh 70m $EXP_NAME $checkpoint_step
done

for checkpoint_step in ${LATE_CHECKPOINT_STEPS[*]}
do
    sbatch run.sh 70m $EXP_NAME $checkpoint_step
done


# sbatch run.sh 70m $EXP_NAME "${MID_CHECKPOINT_STEPS[*]}"
# sbatch run.sh 70m $EXP_NAME "${LATE_CHECKPOINT_STEPS[*]}"

# sbatch run.sh 410m $EXP_NAME "${EARLY_CHECKPOINT_STEPS[*]}"
# sbatch run.sh 410m $EXP_NAME "${MID_CHECKPOINT_STEPS[*]}"
# sbatch run.sh 410m $EXP_NAME "${LATE_CHECKPOINT_STEPS[*]}"

#sbatch run.sh 1.4b $EXP_NAME "${EARLY_CHECKPOINT_STEPS[*]}"
#sbatch run.sh 1.4b $EXP_NAME "${MID_CHECKPOINT_STEPS[*]}"
#sbatch run.sh 1.4b $EXP_NAME "${LATE_CHECKPOINT_STEPS[*]}"

#sbatch run.sh 6.9b $EXP_NAME "${EARLY_CHECKPOINT_STEPS[*]}"
#sbatch run.sh 6.9b $EXP_NAME "${MID_CHECKPOINT_STEPS[*]}"
#sbatch run.sh 6.9b $EXP_NAME "${CUSTOM4[*]}"


# For scaling experiments, only run last checkpoint step
# sbatch run.sh 14m $EXP_NAME  "${LAST_CHECKPOINT_STEP[*]}"
# sbatch run.sh 31m $EXP_NAME  "${LAST_CHECKPOINT_STEP[*]}"
# sbatch run.sh 70m $EXP_NAME  "${LAST_CHECKPOINT_STEP[*]}"
# sbatch run.sh 160m $EXP_NAME "${LAST_CHECKPOINT_STEP[*]}"
# sbatch run.sh 2.8b $EXP_NAME "${LAST_CHECKPOINT_STEP[*]}"
# sbatch run.sh 12b $EXP_NAME  "${LAST_CHECKPOINT_STEP[*]}"





