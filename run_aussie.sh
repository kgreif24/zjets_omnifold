#!/bin/bash

# SLURM submission script for AUSSIE training on Perlmutter.
#
# AUSSIE is single-GPU only (create_graph=True through the step-1 classifier
# breaks under DDP). Each array task runs one ensemble member; the index is
# passed via SLURM_ARRAY_TASK_ID.

#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -q preempt
#SBATCH -J aussie
#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL
#SBATCH -A m3246
#SBATCH -t 0-04:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append

# Job array: one ensemble member per task. Adjust the range as needed.
#SBATCH --array=1-1

#SBATCH -o ./outfiles/%x-%A-%a.out
#SBATCH -e ./outfiles/%x-%A-%a.err

# Set up environment
module load conda
conda activate zfjets

# Set up wandb
export WANDB__SERVICE_WAIT=400
wandb login

# Run AUSSIE training (config path should be edited to match the
# accompanying Omnifold run's config).
python aussie_train.py --config_path ./cli/zjets-v4.yml --index $SLURM_ARRAY_TASK_ID
sleep 120  # allow SLURM to deliver SIGTERM before job exits on preemption
