# README

# Getting Started

## Install 

* `pytorch`: 1.7.0+ (GPU support preferable).

Then,
* `make install`

OR:
* `pip install -r requirements.txt`
* `pip install -e .`

Install as a library:
* `pip install -e .`

### Installing Optuna

Clone hydra from https://github.com/toshihikoyanase/hydra/tree/add-optuna-sweeper.

Install `optuna` under `plugins`. Note that this might be merged into master hydra soon.

### W&B API key

Copy your WandB API key to `wandb_api.key`.
Will be used to login to your dashboard for visualisation. 
Alternatively, you can skip W&B visualisation.

## Unit Tests

`make test`

Run `make help` to see specific make commands.

# Example Code

## Print Current Config

We use [hydra](https://hydra.cc/docs/intro) to handle configs.

```
python main.py --cfg job
```

See `conf/configs` for a detailed list of default configs, and under each folder of `conf` for possible options.

## CIFAR10 with SNFS

```
make cifar10.ERK.SNFS DENSITY=0.2 SEED=0
```

See `outputs/CIFAR10/SNFS` for checkpoints etc. 

## Report Runs

### Visualization Code

TODO: write docs for each python file in `sparselearning/vis_tools/`

# Understanding the config setup

We split configs into various config groups for brevity.

Config groups (example):
* masking
* optimizer
* dataset 
etc.

Hydra allows us to override these either group-wise or globally as described below.
 
## Overrriding specific options / group configs

`python main.py masking=RigL wandb.use=True`

Refer to hydra's documentation for more details.

## Using specific configs

Sometimes, we want to store the specific config of a run with tuned options across mutliple groups (masking, optimizer etc.)

To do so:

* store your config under `specific/`. 
* each YAML file must start with a `# @package _global_` directive. See `specific/` for existing examples. 
* override only what has changed, i.e., donot keep redundant arguments, which the base config (`config.yaml`) already covers.

Syntax:

`python main.py +specific=cifar_wrn_22_2_rigl`