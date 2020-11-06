#!/usr/bin/env bash
#SBATCH --job-name=cifar-wrn-Pruning    # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed
#SBATCH --array=0-2

# Mailing stuff
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vsundar4@wisc.edu
#SBATCH --output=slurm_outputs/log-%x.%A_%a.out

# Job info
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOB_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Time limit: ${SBATCH_TIMELIMIT}"

nvidia-smi

# Conda stuff
module load cuda/10.2 anaconda/wml
source ~/.zshrc
conda activate torch38

# NVIDIA SMI monitoring
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  while true
   do
       nvidia-smi | cat >"slurm_outputs/nvidia-smi-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
       sleep 0.1
   done &
fi

# Start Job here
# Note: we're using the same GPU

python main.py dataset=CIFAR10 optimizer=SGD +specific=cifar_wrn_22_2_pruning exp_name='Pruning' seed=$SLURM_ARRAY_TASK_ID  masking.final_density=0.2 ckpt_interval=1

#python main.py dataset=CIFAR10 optimizer=SGD +specific=cifar_wrn_22_2_pruning exp_name='Pruning_global' seed=$SLURM_ARRAY_TASK_ID  masking.final_density=0.05,0.1,0.2,0.5 use_wandb=True -m
#
#python main.py dataset=CIFAR10 optimizer=SGD +specific=cifar_wrn_22_2_pruning masking.prune_mode=magnitude exp_name='Pruning_local' seed=$SLURM_ARRAY_TASK_ID masking.final_density=0.05,0.1,0.2,0.5 use_wandb=True -m

#wait