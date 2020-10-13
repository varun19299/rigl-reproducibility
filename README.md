# README

# Getting Started

## Install 

* `pytorch`: 1.6.0+
* `pip install -r requirements.txt`

## W&B API key

Copy your WandB API key to `wandb_api.key`.
Will be used to login to your dashboard for visualisation. 
Alternatively, you can skip W&B visualisation.

## Print Current Config

We use [hydra](https://hydra.cc/docs/intro) to handle configs.

```
python main.py --cfg job
```

See `conf/configs` for a detailed list of default configs, and under each folder of `conf` for possible options.

## MNIST with SNFS

```
python main.py
```

See `outputs/SNFS` for checkpoints etc. 

## Understanding the config setup

We split configs into various config groups for brevity.

Config groups (example):
* masking
* optimizer
* dataset 
etc.

Hydra allows us to override these either group-wise or globally as described below.
 
### Overrriding specific options

 
### Overriding group configs

`python main.py masking=RigL`

### Using specific configs

Sometimes, we want to store the specific config of a run with tuned options across mutliple groups (masking, optimizer etc.)

To do so:

* store your config under `specific/`. 
* each YAML file must start with a `# @package _global_` directive. See `specific/` for existing examples. 
* override only what has changed, i.e., donot keep redundant arguments, which the base config (`config.yaml`) already covers.

Syntax:

`python main.py +specific=cifar_wrn_22_2_rigl`

--train_dir "${SCRATCH}/${SLURM_JOB_NAME}_seed_${SLURM_ARRAY_TASK_ID}"