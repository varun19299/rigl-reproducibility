Main Results
============

Pre-trained Models
~~~~~~~~~~~~~~~~~~

All checkpoints can be found `here <https://drive.google.com/drive/folders/17LWYh9mgPUgk4Xe5YKLglzWyWDGk_aYg?usp=sharing/>`_.
Place folders under ``outputs/``.

Commands
~~~~~~~~

The following make command runs all the main results described in our reproducibility report.

.. code-block:: bash

    make cifar10 DENSITY=0.05,0.1,0.2,0.5
    make cifar100 DENSITY=0.05,0.1,0.2,0.5
    make cifar10_tune DENSITY=0.05,0.1,0.2,0.5

Use the ``-n`` flag to see which commands are executed.
Note that these runs are executed sequentially, although we include parallel processes for cifar10 runs of a particular method.
Eg: ``cifar10.Random.RigL`` runs RigL Random for densities ``0.05,0.1,0.2,0.5``, ``seed=0`` in parallel.

It may be preferable to run specific make commands in parallel for this reason. See `make help` for an exhaustive list.

Table of Results
~~~~~~~~~~~~~~~~

Shown for 80% sparsity (20% density) on CIFAR10. For exhaustive results and their analysis refer to our report.

