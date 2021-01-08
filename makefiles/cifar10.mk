## cifar10.ERK.%: Run CIFAR 10 ERK runs. % in RigL, SNFS, SET
cifar10.ERK.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* \
	+specific=cifar10_wrn_22_2_masking seed=$(SEED) exp_name="$*_ERK" \
	masking.density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.2x.ERK.%: Run CIFAR 10 2x ERK runs. % in RigL, SNFS, SET
cifar10.2x.ERK.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* +specific=cifar10_wrn_22_2_masking \
	seed=$(SEED) \
	exp_name="$*_2x_ERK" optimizer.training_multiplier=2 \
	masking.density=$(DENSITY) wandb.project="cifar10" wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.2x.Random.%: Run CIFAR 10 2x Random runs. % in RigL, SNFS, SET
cifar10.2x.Random.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* +specific=cifar10_wrn_22_2_masking \
	seed=$(SEED) masking.sparse_init=random \
	exp_name="$*_2x_Random" optimizer.training_multiplier=2 \
	masking.density=$(DENSITY) wandb.project="cifar10" wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.3x.Random.%: Run CIFAR 10 3x Random runs. % in RigL, SNFS, SET
cifar10.3x.Random.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* +specific=cifar10_wrn_22_2_masking \
	seed=$(SEED) masking.sparse_init=random \
	exp_name="$*_3x_Random" optimizer.training_multiplier=3 \
	masking.density=$(DENSITY) wandb.project="cifar10" wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.Random.%: Run CIFAR 10 Random runs. % in RigL, SNFS, SET
cifar10.Random.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* \
	+specific=cifar10_wrn_22_2_masking seed=$(SEED) exp_name="$*_Random" \
	masking.sparse_init=random masking.density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.Pruning: Pruning runs
cifar10.Pruning:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD \
	+specific=cifar10_wrn_22_2_pruning exp_name='Pruning' seed=$(SEED)  \
	optimizer.warmup_steps=3500 \
	masking.final_density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.Lottery: Lottery runs
cifar10.Lottery: cifar10.Pruning
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=Lottery \
	+specific=cifar10_wrn_22_2_lottery exp_name='Lottery' seed=$(SEED)  \
	masking.density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.Small-Dense: Small-Dense runs
cifar10.Small-Dense:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=Small_Dense \
	+specific=cifar10_wrn_22_2_dense seed=$(SEED)  \
	masking.density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

## cifar10.Static.%: use as cifar10.Static.ERK, cifar10.Static.Random
cifar10.Static.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD \
	+specific=cifar10_wrn_22_2_static seed=$(SEED) exp_name="Static_$*" \
	masking.density=$(DENSITY) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

CIFAR10_DEPS := cifar10.ERK.RigL cifar10.ERK.SNFS cifar10.ERK.SET
CIFAR10_DEPS += cifar10.Random.RigL cifar10.Random.SNFS cifar10.Random.SET
CIFAR10_DEPS += cifar10.2x.ERK.RigL cifar10.2x.ERK.SNFS cifar10.2x.ERK.SET
CIFAR10_DEPS += cifar10.2x.Random.RigL cifar10.2x.Random.SNFS cifar10.2x.Random.SET
CIFAR10_DEPS += cifar10.3x.Random.RigL
CIFAR10_DEPS += cifar10.Pruning cifar10.Lottery cifar10.Small-Dense
CIFAR10_DEPS += cifar10.Static.ERK cifar10.Static.Random

## cifar10: Main CIFAR 10 Runs
cifar10: $(CIFAR10_DEPS)