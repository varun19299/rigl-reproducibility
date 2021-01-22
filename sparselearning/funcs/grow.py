"""
Implements Growth function.

Modifies binary mask to enable gradient flow.
New weights by default 0 and it can
be changed in the function.

Functions have access to the masking object
enabling greater flexibility in designing
custom growth modes.

Signature:
<func>(masking, name, total_regrowth, weight)
"""
from einops import rearrange
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def momentum_growth(
    masking: "Masking",
    name: str,
    total_regrowth: int,
    weight: "Tensor",
) -> "Tensor":
    """
    Grows weights in places where the momentum is largest.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param total_regrowth: amount to re-grow
    :type total_regrowth: int
    :param weight: layer weight
    :type weight: torch.Tensor
    :return: New boolean mask
    :rtype: torch.Tensor
    """
    new_mask = masking.mask_dict[name].data.bool()

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
    total_regrowth: int,
    weight: "Tensor",
) -> "Tensor":
    """
    Grows weights in places where the abs(grad) is largest
    (among present zero'ed weights).

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param total_regrowth: amount to re-grow
    :type total_regrowth: int
    :param weight: layer weight
    :type weight: torch.Tensor
    :return: New boolean mask
    :rtype: torch.Tensor
    """
    new_mask = masking.mask_dict[name].data.bool()

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
    total_regrowth: int,
    weight: "Tensor",
) -> "Tensor":
    """
    Random growth.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param total_regrowth: amount to re-grow
    :type total_regrowth: int
    :param weight: layer weight
    :type weight: torch.Tensor
    :return: New boolean mask
    :rtype: torch.Tensor
    """
    new_mask = masking.mask_dict[name].data.bool()

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
    total_regrowth: int,
    weight: "Tensor",
) -> "Tensor":
    """
    No growth.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param total_regrowth: amount to re-grow
    :type total_regrowth: int
    :param weight: layer weight
    :type weight: torch.Tensor
    :return: New boolean mask
    :rtype: torch.Tensor
    """
    new_mask = masking.mask_dict[name].data.bool()

    return new_mask


def struct_abs_grad_growth(
    masking: "Masking",
    name: str,
    total_regrowth: int,
    weight: "Tensor",
    criterion: Callable = torch.mean,
):
    """
    Performs absolute gradient growth channel-wise.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param total_regrowth: amount to re-grow
    :type total_regrowth: int
    :param weight: layer weight
    :type weight: torch.Tensor
    :param criterion: callable to perform reduction
    :type criterion: Callable
    :return: New boolean mask
    :rtype: torch.Tensor
    """
    new_mask = masking.mask_dict[name].data.bool()

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
