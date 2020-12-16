#!/usr/bin/env bash
#SBATCH --job-name=cifar10-wrn-22-2   # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed
#SBATCH --array=3-3

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
#    python main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
#    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_ERK" \
#    masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m

    python main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_ERK" \
    masking.density=0.0587,0.1145,0.2855 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_Random" \
    masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "SNFS" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=SNFS \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name="SNFS_corrected_ERK" \
    masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=SNFS \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name="SNFS_corrected_Random" \
    masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "SET" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=SET \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name='SET_corrected_ERK' \
    masking.sparse_init=erdos-renyi-kernel masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD masking=SET \
    +specific=cifar10_wrn_22_2_masking seed=$SLURM_ARRAY_TASK_ID exp_name='SET_corrected_Random' \
    masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "Dense" ]; then
  python main.py dataset=CIFAR10 optimizer=SGD masking=Dense \
  +specific=cifar10_wrn_22_2_dense seed=$SLURM_ARRAY_TASK_ID wandb.use=True
fi

if [ ${1} == "Static" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD \
    +specific=cifar10_wrn_22_2_static seed=$SLURM_ARRAY_TASK_ID exp_name="Static_ERK" \
    masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR10 optimizer=SGD \
    +specific=cifar10_wrn_22_2_static seed=$SLURM_ARRAY_TASK_ID exp_name="Static_Random" \
    masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "Small-Dense" ]; then
  python main.py dataset=CIFAR10 optimizer=SGD masking=Small_Dense \
  +specific=cifar10_wrn_22_2_dense seed=$SLURM_ARRAY_TASK_ID \
  masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
fi

if [ ${1} == "Pruning" ]; then
  python main.py dataset=CIFAR10 optimizer=SGD \
  +specific=cifar10_wrn_22_2_pruning exp_name='Pruning' seed=$SLURM_ARRAY_TASK_ID  \
  masking.final_density=0.05,0.1,0.2,0.5 wandb.use=True -m
fi

if [ ${1} == "Lottery" ]; then
  python main.py dataset=CIFAR10 optimizer=SGD masking=Lottery \
  +specific=cifar10_wrn_22_2_lottery exp_name='Lottery' seed=$SLURM_ARRAY_TASK_ID  \
  masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
fi