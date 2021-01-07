#!/usr/bin/env bash
#SBATCH --job-name=cifar100-resnet50    # create a short name for your job

#SBATCH --partition=batch_default   # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=1-20:00:00          # total run time limit (HH:MM:SS)
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
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_ERK" masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m

#    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
#    masking=RigL +specific=cifar100_resnet50_masking \
#    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_ERK" masking.density=0.044,0.088,0.298 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL_Random" masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "RigL-3x" ]; then
  if [ ${2} == "Random" ]; then
    python main.py dataset=CIFAR100 optimizer=SGD masking=RigL \
    optimizer.training_multiplier=3 dataset.max_threads=10 \
    +specific=cifar100_resnet50_masking seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-3x_Random" \
    masking.sparse_init=random masking.density=${3} wandb.use=True
  fi
fi

if [ ${1} == "RigL-SG" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-SG_ERK" \
    masking.redistribution_mode=grad masking.print_FLOPs=true \
    masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-SG_Random" \
    masking.redistribution_mode=grad masking.print_FLOPs=true \
    masking.sparse_init=random masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "RigL-SM" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-SM_ERK" \
    masking.redistribution_mode=momentum masking.print_FLOPs=true \
    masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    # Prioritize 0.1, 0.2 over rest

    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-SM_Random" \
    masking.redistribution_mode=momentum masking.print_FLOPs=true \
    masking.sparse_init=random masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "RigL-reinit" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD masking=RigL_reinit \
    +specific=cifar100_resnet50_masking_reinit seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-reinit_ERK" \
    masking.init_exp_name=RigL-SM_ERK \
    masking.density=0.1,0.2,0.05 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD masking=RigL_reinit \
    +specific=cifar100_resnet50_masking_reinit seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-reinit_Random" \
    masking.init_exp_name=RigL-SM_Random \
    masking.density=0.1,0.2,0.05 wandb.use=True -m
  fi
fi

if [ ${1} == "RigL-struct" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-struct_ERK" \
    masking.sparse_init=struct-erdos-renyi masking.growth_mode=struct-absolute-gradient-mean masking.prune_mode=struct-magnitude-mean \
    masking.density=0.1,0.2 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=RigL +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="RigL-struct_Random" \
    masking.sparse_init=struct-random masking.growth_mode=struct-absolute-gradient-mean masking.prune_mode=struct-magnitude-mean \
    masking.density=0.1,0.2 wandb.use=True -m
  fi
fi

if [ ${1} == "SNFS" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=SNFS +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="SNFS_ERK" masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=SNFS +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name="SNFS_Random" masking.sparse_init=random masking.density=0.1,0.2,0.05,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "SET" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=SET +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name='SET_ERK' masking.sparse_init=erdos-renyi-kernel masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    masking=SET +specific=cifar100_resnet50_masking \
    seed=$SLURM_ARRAY_TASK_ID exp_name='SET_Random' masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "Dense" ]; then
  python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
  masking=Dense +specific=cifar100_resnet50_dense \
  exp_name="Dense" seed=$SLURM_ARRAY_TASK_ID wandb.use=True
fi

if [ ${1} == "Static" ]; then
  if [ ${2} == "ERK" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    +specific=cifar100_resnet50_static \
    seed=$SLURM_ARRAY_TASK_ID exp_name="Static_ERK" masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi

  if [ ${2} == "Random" ]; then
    python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
    +specific=cifar100_resnet50_static \
    seed=$SLURM_ARRAY_TASK_ID exp_name="Static_Random" masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
  fi
fi

if [ ${1} == "Small-Dense" ]; then
  python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
  masking=Small_Dense +specific=cifar100_resnet50_dense \
  seed=$SLURM_ARRAY_TASK_ID masking.density=0.05,0.1,0.2,0.5 wandb.use=True -m
fi

if [ ${1} == "Pruning" ]; then
  python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD \
  +specific=cifar100_resnet50_pruning exp_name='Pruning' \
  seed=$SLURM_ARRAY_TASK_ID  masking.final_density=0.1,0.2,0.05,0.5 wandb.use=True -m
fi

if [ ${1} == "Lottery" ]; then
  python main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD masking=Lottery \
  +specific=cifar100_resnet50_lottery exp_name='Lottery' \
  seed=$SLURM_ARRAY_TASK_ID  masking.density=0.2,0.5 wandb.use=True -m
fi
#wait