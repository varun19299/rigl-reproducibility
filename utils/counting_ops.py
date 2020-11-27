import logging
from models import registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import CosineDecay
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


def wrn_22_2_FLOPs(sparse_init: str = "random", density: float = 0.2) -> int:
    model_class, args = registry["wrn-22-2"]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mask = Masking(optimizer, decay, sparse_init=sparse_init, density=density)
    mask.add_module(model)

    return get_FLOPs(mask, torch.rand(1, 3, 32, 32))


if __name__ == "__main__":
    dense_FLOPs = wrn_22_2_FLOPs(density=1.0)
    print(f"WRN-22-2 Dense FLOPS: {dense_FLOPs:,} \n")

    for density in [0.05, 0.1, 0.2, 0.5]:
        Random_FLOPs = wrn_22_2_FLOPs("random", density)
        ER_FLOPs = wrn_22_2_FLOPs("erdos-renyi", density)
        ERK_FLOPs = wrn_22_2_FLOPs("erdos-renyi-kernel", density)

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
