import logging
import torch
from torch import nn
from typing import TYPE_CHECKING
import utils.micronet_challenge.counting as counting
from utils.micronet_challenge.counting import count_ops

if TYPE_CHECKING:
    from utils.typing_alias import *


def get_FLOPS(masking: "Masking", input_tensor: "Tensor", param_size: int = 32):
    total_FLOPS = 0

    activation_dict = get_activations_dict(masking.module, input_tensor)

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
        sparsity = (weight.data == 0).sum().int() / weight.numel()
        print(name, activation_dict[name].shape, sparsity)
        param_count, n_mult, n_add = counting.count_ops(layer_op, sparsity, param_size)

        total_FLOPS += n_mult + n_add

    return total_FLOPS.int().item()


def get_activations_dict(net: "nn.Module", input_tensor: "Tensor"):
    """
    Find (pre)activation dict for every possible module in net
    """
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


if __name__ == "__main__":
    from sparselearning.models import WideResNet, registry
    from sparselearning.core import Masking
    from sparselearning.funcs.decay import CosineDecay
    from torch import optim
    from torchsummary import summary

    model = WideResNet(*registry["wrn-22-2"][1])

    activation_dict = get_activations_dict(model, torch.rand(1, 3, 32, 32))

    print(activation_dict["block1.layer.0.relu1"].shape)

    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mask = Masking(optimizer, decay, sparse_init="erdos-renyi-kernel", density=0.1)
    # mask = Masking(optimizer, decay, sparse_init="random", density=0.1)
    mask.add_module(model)
    total_FLOPS = get_FLOPS(mask, torch.rand(1, 3, 32, 32))

    print(f"{total_FLOPS:,}")
