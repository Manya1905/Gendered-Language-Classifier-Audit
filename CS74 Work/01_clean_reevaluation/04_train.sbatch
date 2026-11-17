#!/bin/bash
# E1 step 4 on Dartmouth Discovery: trains the 3 seeds at the same time (a job array).
#
# Submit from ~/cs74/01_clean_reevaluation with:   sbatch 04_train.sbatch
# Check on it with:             squeue -u $USER
#
#SBATCH --job-name=e1_train
#SBATCH --partition=a5000,gpu_preempt     # whichever has a free GPU first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2                       # 3 jobs: one per seed
#SBATCH --output=logs/e1_train_%A_%a.out
#SBATCH --error=logs/e1_train_%A_%a.err

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module load python/anaconda3
export PYTHONNOUSERSITE=1                 # ignore the old CS72 packages in ~/.local
source activate cs74

cd ~/cs74/01_clean_reevaluation
echo "Job $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID) on $SLURM_NODELIST, seed $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU - stopping'; print('GPU OK')" || exit 1

python 04_train.py --seed "$SEED"
echo "Finished at $(date)"
