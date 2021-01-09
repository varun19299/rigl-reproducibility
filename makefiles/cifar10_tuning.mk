define cifar10_ERK_lr_specific
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
	+specific=cifar10_wrn_22_2_grid_lr exp_name='RigL_ERK_grid_lr' \
	masking.density=$(DENSITY) masking.prune_rate=$(1) masking.interval=$(2) \
	optimizer.lr=0.1,0.05,0.01,0.005 wandb.use=$(USE_WANDB) -m
endef

define cifar10_Random_lr_specific
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=RigL \
	+specific=cifar10_wrn_22_2_grid_lr exp_name='RigL_Random_grid_lr' \
	masking.density=$(DENSITY) masking.prune_rate=$(1) masking.interval=$(2) \
	masking.sparse_init=random optimizer.lr=0.1,0.05,0.01,0.005 wandb.use=$(USE_WANDB) -m
endef

## cifar10_lr_grid: Run LR tuning, grid search
cifar10_lr_grid:
	$(call cifar10_ERK_lr_specific,0.3, 100)
	$(call cifar10_ERK_lr_specific,0.4, 200)
	$(call cifar10_ERK_lr_specific,0.4, 500)
	$(call cifar10_ERK_lr_specific,0.5, 750)

	$(call cifar10_Random_lr_specific,0.3, 100)
	$(call cifar10_Random_lr_specific,0.4, 200)
	$(call cifar10_Random_lr_specific,0.4, 500)
	$(call cifar10_Random_lr_specific,0.5, 750)

## cifar10_ERK_alpha_deltaT_optuna: tune alpha, deltaT with optuna. Supply DENSITY.
cifar10_ERK_alpha_deltaT_optuna:
	${PYTHON} main.py hydra/sweeper=optuna dataset=CIFAR10 optimizer=SGD masking=RigL \
	+specific=cifar10_wrn_22_2_optuna exp_name='RigL_ERK_optuna_multiseed' \
    masking.density=$(DENSITY) multi_seed='[0,1,2]' \
    'masking.prune_rate=interval(0.1,0.6)' 'masking.interval=range(50,1000,50)' \
    wandb.use=$(USE_WANDB) -m

## cifar10_Random_alpha_deltaT_optuna: tune alpha, deltaT with optuna. Supply DENSITY.
cifar10_Random_alpha_deltaT_optuna:
	${PYTHON} main.py hydra/sweeper=optuna dataset=CIFAR10 optimizer=SGD masking=RigL \
	+specific=cifar10_wrn_22_2_optuna exp_name='RigL_Random_optuna_multiseed' \
    masking.density=$(DENSITY) multi_seed='[0,1,2]' \
    'masking.prune_rate=interval(0.1,0.6)' 'masking.interval=range(50,1000,50)' \
    masking.sparse_init=random wandb.use=$(USE_WANDB) -m

## cifar10_tune: Run all tuning experiments for CIFAR10
cifar10_tune: cifar10_lr_grid cifar10_ERK_alpha_deltaT_optuna cifar10_Random_alpha_deltaT_optuna