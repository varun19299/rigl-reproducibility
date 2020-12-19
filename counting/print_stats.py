from functools import partial

from counting import model_inference_FLOPs
from counting.inference_train_FLOPs import RigL_train_FLOPs, SNFS_train_FLOPs, SET_train_FLOPs, Pruning_train_FLOPs
from sparselearning.funcs.decay import MagnitudePruneDecay
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.typing_alias import *

def print_stats(model_name: str, input_size: "Tuple" = (1, 3, 32, 32)):
    _model_FLOPs = partial(model_inference_FLOPs, model_name=model_name, input_size=input_size)

    dense_FLOPs = _model_FLOPs(density=1.0)
    dense_train_FLOPs = 3 * dense_FLOPs  # gradient of param and activation
    print(f"Resnet50 Dense FLOPS: {dense_FLOPs:,} \n")

    # Pruning & Masking
    total_steps = 87891
    T_max = 65918
    interval = 100
    # Pruning
    T_start = 700

    for density in [0.1, 0.2]:
        Random_FLOPs = _model_FLOPs("random", density)
        ER_FLOPs = _model_FLOPs("erdos-renyi", density)
        ERK_FLOPs = _model_FLOPs("erdos-renyi-kernel", density)

        pruning_decay = MagnitudePruneDecay(
            final_sparsity=1 - density, T_max=T_max, T_start=T_start, interval=interval
        )

        print(
            f"Random Density: {density} Inference FLOPs:{Random_FLOPs:,} Proportion:{Random_FLOPs / dense_FLOPs:.4f}"
        )
        print(
            f"ER Density: {density} Inference FLOPs:{ER_FLOPs:,} Proportion:{ER_FLOPs / dense_FLOPs:.4f}"
        )
        print(
            f"ERK Density: {density} Inference FLOPs:{ERK_FLOPs:,} Proportion:{ERK_FLOPs / dense_FLOPs:.4f}"
        )

        print("\n")

        for sparse_FLOPs, init_name in zip(
            [Random_FLOPs, ERK_FLOPs], ["Random", "ERK"]
        ):
            set_train_FLOPs = SET_train_FLOPs(sparse_FLOPs, dense_FLOPs, interval)
            snfs_train_FLOPs = SNFS_train_FLOPs(sparse_FLOPs, dense_FLOPs, interval)
            rigl_train_FLOPs = RigL_train_FLOPs(sparse_FLOPs, dense_FLOPs, interval)

            print(
                f"SET {init_name.capitalize()} Density: {density} Train FLOPs:{set_train_FLOPs:,} Proportion:{set_train_FLOPs / dense_train_FLOPs:.4f}"
            )
            print(
                f"SNFS {init_name.capitalize()} Density: {density} Train FLOPs:{snfs_train_FLOPs:,} Proportion:{snfs_train_FLOPs / dense_train_FLOPs:.4f}"
            )
            print(
                f"RigL {init_name.capitalize()} Density: {density} Train FLOPs:{rigl_train_FLOPs:,} Proportion:{rigl_train_FLOPs / dense_train_FLOPs:.4f}"
            )

            print("\n")

        pruning_train_FLOPs = Pruning_train_FLOPs(
            dense_FLOPs, pruning_decay, total_steps=total_steps
        )
        print(
            f"Pruning Density: {density} Train FLOPs:{pruning_train_FLOPs:,} Proportion:{pruning_train_FLOPs / dense_train_FLOPs:.4f}"
        )

        print("-----------\n")

if __name__ == "__main__":
    print_stats("wrn-22-2")
    print_stats("resnet50")