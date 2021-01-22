Getting Started
==============

Install
~~~~~~~~

First install:

* ``python3.8``
* ``pytorch``: 1.7.0+ (GPU support preferable).

Then run:

* ``make install``

W&B API key
~~~~~~~~

Copy your WandB API key to ``wandb_api.key``.
Will be used to login to your dashboard for visualisation.
Alternatively, you can skip W&B visualisation,
and set ``wandb.use=False`` while running the python code or ``USE_WANDB=False`` while running make commands.

Unit Tests
~~~~~~~~

`make test`. Run `make help` to see specific make commands.