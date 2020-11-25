# Define macros
PYTHON := python
HYDRA_FLAGS = -m
SEED = 0

.PHONY: help rigl

.DEFAULT: help
help:
	@echo "make venv"
	@echo "       prepare development environment, use only once"
	@echo "make lint"
	@echo "       run pylint"
	@echo "make run"
	@echo "       run project"

rigl:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=RigL +specific=cifar_wrn_22_2_masking seed=${SEED} exp_name="RigL_ERK" masking.density=0.05,0.1,0.2,0.5 wandb.use=True ${HYDRA_FLAGS}

	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=RigL +specific=cifar_wrn_22_2_masking seed=${SEED} exp_name="RigL_Random" masking.sparse_init=random masking.density=0.05,0.1,0.2,0.5 wandb.use=True ${HYDRA_FLAGS}

cifar10: rigl