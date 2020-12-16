#!/usr/bin/env bash
# Run as sbatch <script name> <density>

#SBATCH --job-name=cifar-wrn-hyperparam-optuna   # create a short name for your job
#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=4-05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed
#SBATCH --array=0-1

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

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    python main.py hydra/sweeper=optuna dataset=CIFAR10 optimizer=SGD masking=RigL +specific=cifar10_wrn_22_2_optuna exp_name='RigL_ERK_optuna_multiseed' \
    masking.density=${1} multi_seed='[0,1,2]' \
    'masking.prune_rate=interval(0.1,0.6)' 'masking.interval=range(50,1000,50)' wandb.use=True -m
fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  python main.py hydra/sweeper=optuna dataset=CIFAR10 optimizer=SGD masking=RigL +specific=cifar10_wrn_22_2_optuna exp_name='RigL_Random_optuna_multiseed' \
  masking.density=${1} multi_seed='[0,1,2]' \
  'masking.prune_rate=interval(0.1,0.6)' 'masking.interval=range(50,1000,50)' masking.sparse_init=random wandb.use=True -m
fi