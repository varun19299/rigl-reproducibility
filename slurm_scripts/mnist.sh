#!/usr/bin/env bash
#SBATCH --job-name=mnist-test    # create a short name for your job

#SBATCH --partition=wacc         # use batch_default, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:gtx1080:1     # GPU needed

# Mailing stuff
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=vsundar4@wisc.edu
#SBATCH --output=slurm_outputs/log-%x.%j.out

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
while true
 do
     nvidia-smi 2>&1 | tee "slurm_outputs/nvidia-smi-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
     sleep 0.1
 done &

# Start Job here
python main.py