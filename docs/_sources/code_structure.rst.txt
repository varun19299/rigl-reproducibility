Code Structure
==============

This section may be useful if you desire to extend this code base or understand its structure.
``main.py`` is the python file used for training-evaluating, and the ``make`` commands serve as a wrapper for it.

Print current config
~~~~~~~~~~~~~~~~~~~~

We use `hydra <https://hydra.cc/docs/intro/>`_ to handle configs.

.. code-block:: bash

    python main.py --cfg job

See ``conf/configs`` for a detailed list of default configs, and under each folder of ``conf`` for possible options.

Understanding the config setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We split configs into various config groups for brevity.

Config groups (example):

    * masking
    * optimizer
    * dataset
etc.

Hydra allows us to override these either group-wise or globally as described below.

Overrriding options / group configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    python main.py masking=RigL wandb.use=True

Refer to hydra's documentation for more details.

Exhaustive config options
~~~~~~~~~~~~~~~~~~~~~~~~~

See ``conf/config.yaml`` and the defaults it uses (eg: ``dataset: CIFAR10``, ``optimizer: SGD``, etc.).

Using specific configs
~~~~~~~~~~~~~~~~~~~~~~

Sometimes, we want to store the specific config of a run with tuned options across mutliple groups (masking, optimizer etc.)

To do so:

* store your config under ``specific/``.
* each YAML file must start with a ``# @package _global_`` directive. See ``specific/`` for existing examples.
* override only what has changed, i.e., donot keep redundant arguments, which the base config (``config.yaml``) already covers.

Syntax:

.. code-block:: bash

    python main.py +specific=cifar_wrn_22_2_rigl