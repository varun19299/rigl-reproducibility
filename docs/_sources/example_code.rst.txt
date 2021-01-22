Example Code
==============

Train WideResNet-22-2 with RigL on CIFAR10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. highlight:: sh

.. code-block:: bash

    make cifar10.ERK.RigL DENSITY=0.2 SEED=0


Change ``DENSITY`` incase you want to use a different density (1 - sparsity) level.
See ``outputs/CIFAR10/RigL_ERK/0.2/`` for checkpoints etc.

Train ResNet-50 with SNFS on CIFAR100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make cifar100.ERK.SNFS DENSITY=0.2 SEED=0


See ``outputs/CIFAR100/SNFS_ERK/0.2`` for checkpoints etc.

Evaluate WideResNet-22-2 with RigL on CIFAR10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Either train WRN-22-2 with RigL as described above, or download checkpoints from `here <https://drive.google.com/drive/folders/1f_q5pm5DR2a3GTGIa-xagWU3Nici8Lq-?usp=sharing/>`_.
Place under ``outputs/CIFAR10/RigL_ERK/0.2/+specific=cifar10_wrn_22_2_masking,seed=0``.

.. code-block:: bash

    make cifar10.ERK.RigL DENSITY=0.2 SEED=0


Evaluate ResNet-50 with SNFS on CIFAR100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Either train ResNet-50 with SNFS as described above, or download checkpoints from `here <https://drive.google.com/drive/folders/1iSooN25SiAsNWF4uKgYnU-9fU-wUp0Hc?usp=sharing/>`_.
Place under ``outputs/CIFAR100/SNFS_ERK/0.2/+specific=cifar100_resnet50_masking,seed=0``.

.. code-block:: bash

    make cifar10.ERK.RigL DENSITY=0.2 SEED=0