#!/usr/bin/env bash
#SBATCH --job-name=cifar10-wrn-22-2-specific   # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed
#SBATCH --array=1-2

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
module load cuda/11.0 anaconda/wml
source ~/.zshrc
conda activate torch1.7_py38

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

if [ ${1} == "RigL" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
    +specific=cifar10_wrn_22_2_optuna seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_ERK" \
    masking.prune_rate=${4} masking.interval=${5} \
    masking.density=${3} wandb.use=True
  fi

  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
    +specific=cifar10_wrn_22_2_optuna seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_Random" \
    masking.prune_rate=${4} masking.interval=${5} \
    masking.density=${3} wandb.use=True
  fi
fi
