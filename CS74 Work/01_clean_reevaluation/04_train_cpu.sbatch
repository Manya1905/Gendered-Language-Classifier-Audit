#!/bin/bash
# E1 step 4, CPU version (backup for when GPUs are busy). Same training, no GPU.
# CS72's original model was trained on CPU too (1 h 44 min on 4 CPUs).
# Saves to output_cpu/ so it can never overwrite the GPU runs in output/.
#
# Submit from ~/cs74/01_clean_reevaluation with:   sbatch 04_train_cpu.sbatch
#
#SBATCH --job-name=e1_train_cpu
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-2                       # 3 jobs: one per seed
#SBATCH --output=logs/e1_train_cpu_%A_%a.out
#SBATCH --error=logs/e1_train_cpu_%A_%a.err

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module load python/anaconda3
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # let PyTorch use all 8 CPUs
source activate cs74

cd ~/cs74/01_clean_reevaluation
echo "Job $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID) on $SLURM_NODELIST, seed $SEED, CPU"
python 04_train.py --seed "$SEED" --out-dir output_cpu
echo "Finished at $(date)"
