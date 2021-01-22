import logging

from sparselearning.counting.helper import get_pre_activations_dict
import torch
from torch import nn
from typing import TYPE_CHECKING
from sparselearning.counting import micronet_challenge as counting

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


@torch.no_grad()
def get_inference_FLOPs(
    masking: "Masking", input_tensor: "Tensor", param_size: int = 32
) -> int:
    """
    Returns total FLOPs consumed for a forward pass (inference FLOPs).
    Assumes support for sparse convolutions, sparse dense layers.

    :param masking: Masking instance, a wrapper on boolean masks
    :type masking: Masking
    :param input_tensor: Input to model, single input supported
    :type input_tensor: torch.Tensor
    :param param_size: bits used for floating point operations (default 32)
    :type param_size: int
    :return: total FLOPs consumed for a forward pass
    :rtype: int
    """
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
