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