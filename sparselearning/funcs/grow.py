"""
Implements Growth function.

Modifies binary mask to enable gradient flow.
New weights by default 0 and it can
be changed in the function.

Functions have access to the masking object
enabling greater flexibility in designing
custom growth modes.

Signature:
<func>(masking, name, new_mask, total_regrowth, weight: "Tensor")
"""
from einops import rearrange
from functools import partial
import math
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def momentum_growth(
    masking: "Masking",
    name: str,
    new_mask: "Tensor",
    total_regrowth: int,
    weight: "Tensor",
):
    """Grows weights in places where the momentum is largest.



    Operates in-place manner, with new_mask modified.


    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        new_mask    The binary mask. 1s indicated active weights.
                    This binary mask has already been pruned in the
                    pruning step that preceeds the growth step.

        total_regrowth    This variable determines the number of
                    parameters to regrowtn in this function.
                    It is automatically determined by the
                    redistribution function and algorithms
                    internal to the sparselearning library.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.

    Access to optimizer:
        masking.optimizer

    Access to momentum/Adam update:
        masking.get_momentum_for_weight(weight)

    Accessible global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.stats.nonzeros_dict[name]
        Zero count of layer:
            masking.stats.zeros_dict[name]
        Redistribution proportion:
            masking.stats.variance_dict[name]
        Number of items removed through pruning:
            masking.stats.removed_dict[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.stats.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.stats.total_zero = 0
        Total number of parameters removed in pruning:
            masking.stats.total_removed = 0
    """
    momentum = masking.get_momentum_for_weight(weight)
    if momentum.dtype == torch.float16:
        momentum = momentum * (new_mask == 0).half()
    else:
        momentum = momentum * (new_mask == 0).float()
    y, idx = torch.sort(torch.abs(momentum).flatten(), descending=True)
    new_mask.data.view(-1)[idx[: int(total_regrowth)]] = 1.0

    return new_mask


def abs_grad_growth(
    masking: "Masking",
    name: str,
    new_mask: "Tensor",
    total_regrowth: int,
    weight: "Tensor",
):
    """
    Grows weights in places where the abs(grad) is largest.
    (among present zero'ed weights)

    Operates in-place manner, with new_mask modified.

    :param masking: Masking instance
    :param name: layer name
    :param new_mask: output boolean tensor
    :param total_regrowth: amount to re-grow
    :param weight: layer weight
    :return:
    """
    # If dense, skip
    n = (new_mask == 0).sum().item()
    if n == 0:
        return new_mask

    grad = weight.grad
    if grad.dtype == torch.float16:
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[: int(total_regrowth)]] = 1.0

    # init new weights to 0
    weight.data.view(-1)[idx[: int(total_regrowth)]] = 0.0

    return new_mask


def random_growth(
    masking: "Masking",
    name: str,
    new_mask: "Tensor",
    total_regrowth: int,
    weight: "Tensor",
):
    """
    Random growth

    :param masking: Masking instance
    :param name: layer name
    :param new_mask: output boolean tensor
    :param total_regrowth: amount to re-grow
    :param weight: layer weight
    :return:
    """
    # If dense, skip
    n = (new_mask == 0).sum().item()
    if n == 0:
        return new_mask
    expeced_growth_probability = total_regrowth / n
    new_weights = torch.zeros_like(new_mask).bool()
    new_weights[new_mask == 0] = (
        torch.rand_like(new_weights[new_mask == 0].float()) < expeced_growth_probability
    )
    new_mask = new_mask.bool() | new_weights.bool()

    # init new weights to 0
    weight.data[new_weights == 1] = 0.0
    weight.data[new_mask == 0] = 0.0

    return new_mask


def no_growth(
    masking: "Masking",
    name: str,
    new_mask: "Tensor",
    total_regrowth: int,
    weight: "Tensor",
):
    """
    No growth

    :param masking: Masking instance
    :param name: layer name
    :param new_mask: output boolean tensor
    :param total_regrowth: amount to re-grow
    :param weight: layer weight
    :return:
    """
    return new_mask


def struct_abs_grad_growth(
    masking: "Masking",
    name: str,
    new_mask: "Tensor",
    total_regrowth: int,
    weight: "Tensor",
    criterion: Callable = torch.mean,
):
    """
    Performs absolute gradient growth channel-wise

    :param masking: Masking instance
    :param name: layer name
    :param new_mask: output boolean tensor
    :param total_regrowth: amount to re-grow
    :param weight: layer weight
    :param criterion: callable to perform reduction
    :return:
    """

    # If dense, skip
    n = (new_mask == 0).sum().item()
    if n == 0:
        return new_mask

    grad = weight.grad
    if grad.dtype == torch.float16:
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()

    c_in, c_out, h, w = weight.shape
    kernel_size = h * w

    reduced = criterion(rearrange(grad, "c_in c_out h w -> c_in c_out (h w)"), axis=-1)

    y, idx = torch.sort(torch.abs(reduced).flatten(), descending=True)

    new_mask.data.view(-1, h, w)[idx[: int(total_regrowth / kernel_size)], :, :] = 1.0

    # init new weights to 0
    weight.data.view(-1, h, w)[idx[: int(total_regrowth / kernel_size)], :, :] = 0.0

    return new_mask


registry = {
    "absolute-gradient": abs_grad_growth,
    "momentum": momentum_growth,
    "none": no_growth,
    "random": random_growth,
    "struct-absolute-gradient-mean": partial(
        struct_abs_grad_growth, criterion=torch.mean
    ),
    "struct-absolute-gradient-min": partial(
        struct_abs_grad_growth, criterion=lambda x, **kwargs: torch.min(x, **kwargs)[0]
    ),
}
