import logging
from models import registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import CosineDecay, MagnitudePruneDecay
import torch
from torch import nn, optim
from typing import TYPE_CHECKING
import utils.micronet_challenge.counting as counting

if TYPE_CHECKING:
    from utils.typing_alias import *


def get_FLOPs(masking: "Masking", input_tensor: "Tensor", param_size: int = 32):
    total_FLOPs = 0

    activation_dict = get_pre_activations_dict(masking.module, input_tensor)

    for name, module in masking.module.named_modules():
        layer_op = None

        if isinstance(module, nn.Conv2d):
            has_bias = isinstance(module.bias, torch.Tensor)
            assert (
                module.kernel_size[0] == module.kernel_size[1]
            ), "Counting module requires square kernels."

            padding_type = (
                "same" if module.padding[0] == module.kernel_size[0] // 2 else "valid"
            )
            c_in, c_out, k_w, k_h = module.weight.shape
            k_shape = [k_w, k_h, c_in, c_out]  # alas Tensorflow!

            # Depth Separable convolution
            if module.groups == module.in_channels:
                layer_op = counting.DepthWiseConv2D(
                    activation_dict[name].shape[2],
                    k_shape,
                    module.stride,
                    padding_type,
                    has_bias,
                    "relu",
                )
            else:
                layer_op = counting.Conv2D(
                    activation_dict[name].shape[2],
                    k_shape,
                    module.stride,
                    padding_type,
                    has_bias,
                    "relu",
                )

        elif isinstance(module, nn.Linear):
            has_bias = isinstance(module.bias, torch.Tensor)
            layer_op = counting.FullyConnected(module.weight.shape, has_bias, "relu")

        if not layer_op:
            continue

        weight = module.weight
        sparsity = (weight.data == 0).sum().item() / weight.numel()
        param_count, n_mult, n_add = counting.count_ops(layer_op, sparsity, param_size)

        logging.debug(
            f"{name}: shape {weight.shape} params: {param_count} sparsity: {sparsity} FLOPs: {n_mult+ n_add}"
        )
        total_FLOPs += n_mult + n_add

    total_FLOPs = int(total_FLOPs)
    logging.debug(f"Total FLOPs: {total_FLOPs}")
    return total_FLOPs


def get_pre_activations_dict(net: "nn.Module", input_tensor: "Tensor"):
    """
    Find (pre)activation dict for every possible module in net
    """
    # TODO: this function invokes the warning
    # torch/nn/modules/container.py:434: UserWarning: Setting attributes on ParameterList is not supported.
    # warnings.warn("Setting attributes on ParameterList is not supported.")
    # Why?

    activation_dict = {}

    def _get_activation(name):
        def hook(model, input, output):
            # activation_dict[name] = output.detach()
            activation_dict[name] = input[0].detach()

        return hook

    for name, module in net.named_modules():
        module.register_forward_hook(_get_activation(name))

    net(input_tensor)

    return activation_dict


def wrn_22_2_FLOPs(sparse_init: str = "random", density: float = 0.2,) -> int:
    model_class, args = registry["wrn-22-2"]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mask = Masking(optimizer, decay, sparse_init=sparse_init, density=density)
    mask.add_module(model)

    return get_FLOPs(mask, torch.rand(1, 3, 32, 32))


def RigL_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return (2 * sparse_FLOPs + dense_FLOPs + 3 * sparse_FLOPs * mask_interval) / (
        mask_interval + 1
    )


def SNFS_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return 2 * sparse_FLOPs + dense_FLOPs


def SET_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return 3 * sparse_FLOPs


def Pruning_FLOPs(
    dense_FLOPs: int, decay: MagnitudePruneDecay, total_steps: int = 87891
):
    avg_sparsity = 0.0
    for i in range(0, total_steps):
        avg_sparsity += decay.cumulative_sparsity(i)

    avg_sparsity /= total_steps

    return 3 * dense_FLOPs * (1 - avg_sparsity)


# TODO: Pruning calculation
# TODO: RigL, SNFS, SET calculation

if __name__ == "__main__":
    dense_FLOPs = wrn_22_2_FLOPs(density=1.0)
    dense_train_FLOPs = 3 * dense_FLOPs  # gradient of param and activation
    print(f"WRN-22-2 Dense FLOPS: {dense_FLOPs:,} \n")

    # Pruning & Masking
    total_steps = 87891
    T_max = 65918
    interval = 100
    # Pruning
    T_start = 700

    for density in [0.1, 0.2]:
        Random_FLOPs = wrn_22_2_FLOPs("random", density)
        ER_FLOPs = wrn_22_2_FLOPs("erdos-renyi", density)
        ERK_FLOPs = wrn_22_2_FLOPs("erdos-renyi-kernel", density)

        pruning_decay = MagnitudePruneDecay(
            final_sparsity=1 - density, T_max=T_max, T_start=T_start, interval=interval
        )

        print(
            f"Random Density: {density} Inference FLOPs:{Random_FLOPs:,} Proportion:{Random_FLOPs/dense_FLOPs:.4f}"
        )
        print(
            f"ER Density: {density} Inference FLOPs:{ER_FLOPs:,} Proportion:{ER_FLOPs/dense_FLOPs:.4f}"
        )
        print(
            f"ERK Density: {density} Inference FLOPs:{ERK_FLOPs:,} Proportion:{ERK_FLOPs/dense_FLOPs:.4f}"
        )

        print("\n")

        for sparse_FLOPs, init_name in zip(
            [Random_FLOPs, ERK_FLOPs], ["Random", "ERK"]
        ):
            set_train_FLOPs = SET_FLOPs(sparse_FLOPs, dense_FLOPs, interval)
            snfs_train_FLOPs = SNFS_FLOPs(sparse_FLOPs, dense_FLOPs, interval)
            rigl_train_FLOPs = RigL_FLOPs(sparse_FLOPs, dense_FLOPs, interval)

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

        pruning_train_FLOPs = Pruning_FLOPs(
            dense_FLOPs, pruning_decay, total_steps=total_steps
        )
        print(
            f"Pruning Density: {density} Train FLOPs:{pruning_train_FLOPs:,} Proportion:{pruning_train_FLOPs / dense_train_FLOPs:.4f}"
        )

        print("-----------\n")
