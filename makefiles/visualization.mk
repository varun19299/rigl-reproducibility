## vis.main_result.cifar10: tabulate cifar10 results
vis.main_result.cifar10:
	${PYTHON} visualization/main_results.py wandb.project=cifar10 dataset=CIFAR10

## vis.main_result.cifar100: tabulate cifar100 results
vis.main_result.cifar100:
	${PYTHON} visualization/main_results.py wandb.project=cifar100 dataset=CIFAR100

## vis.tuning.alpha_deltaT: alpha DeltaT tuning with optuna contour plots
vis.tuning.alpha_deltaT:
	${PYTHON} visualization/alpha_deltaT.py wandb.project="cifar10 optuna multiseed" dataset=CIFAR10

## vis.tuning.lr: LR tuning with grid search plots
vis.tuning.lr:
	${PYTHON} visualization/lr_tuning.py wandb.project="cifar10 grid lr" dataset=CIFAR10

## vis.erk_vs_random_FLOPs: plot ERK vs Random performance for fixed FLOPs
vis.erk_vs_random_FLOPs:
	${PYTHON} visualization/erk_vs_random_FLOPs.py

## vis.redist_inference_FLOPs: plot RigL-Random, SG, SM, ERK inference FLOPs vs train steps, on CIFAR100, 80% sparsity.
vis.redist_inference_FLOPs:
	${PYTHON} visualization/redist_inference_plot.py wandb.project=cifar100 dataset=CIFAR100

## vis.inference_stats: estimate train and eval FLOPs (assuming final density is similar to initial)
vis.inference_stats:
	${PYTHON} sparselearning/counting/print_stats.py

VIS_DEPS := vis.main_result.cifar10 vis.main_result.cifar100
VIS_DEPS += vis.tuning.alpha_deltaT vis.tuning.lr
VIS_DEPS += vis.erk_vs_random_FLOPs vis.redist_inference_FLOPs
## vis: All visualizations
vis: $(VIS_DEPS)