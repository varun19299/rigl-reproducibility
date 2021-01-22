"""
Implements Pruning function.

Modifies binary mask to prevent gradient flow.

Functions have access to the masking object
enabling greater flexibility in designing
custom prune modes.

Signature:
<func>(masking, mask, weight, name)
"""
from einops import rearrange
from functools import partial
import math
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def magnitude_prune(
    masking: "Masking", mask: "Tensor", weight: "Tensor", name: str
) -> "Tensor":
    """
    Prunes the weights with smallest magnitude.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param mask: layer mask
    :type mask: torch.Tensor
    :param weight: layer weight
    :type weight: torch.Tensor
    :param name: layer name
    :type name: str
    :return: pruned mask
    :rtype: torch.Tensor
    """
    num_remove = math.ceil(
        masking.name2prune_rate[name] * masking.stats.nonzeros_dict[name]
    )
    if num_remove == 0.0:
        return mask
    num_zeros = masking.stats.zeros_dict[name]
    k = num_zeros + num_remove

    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


def global_magnitude_prune(masking: "Masking") -> int:
    """
    Global Magnitude (L1) pruning. Modifies in-place.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :return: number of weights removed
    :rtype: int
    """
    tokill = math.ceil(masking.prune_rate * masking.baseline_nonzero)
    if tokill <= 0:
        return 0
    total_removed = 0
    prev_removed = 0

    if tokill:
        increment = masking.increment
        tries_before_breaking = 10
        tries = 0

        while abs(total_removed - tokill) > tokill * masking.tolerance:
            total_removed = 0
            for name, weight in masking.module.named_parameters():
                if name not in masking.mask_dict:
                    continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.stats.nonzeros_dict[name] - remain

            if prev_removed == total_removed:
                tries += 1
                if tries == tries_before_breaking:
                    break
            else:
                tries = 0

            prev_removed = total_removed
            if total_removed > tokill * (1.0 + masking.tolerance):
                masking.prune_threshold *= 1.0 - increment
                increment *= 0.99
            elif total_removed < tokill * (1.0 - masking.tolerance):
                masking.prune_threshold *= 1.0 + increment
                increment *= 0.99

        for name, weight in masking.module.named_parameters():
            if name not in masking.mask_dict:
                continue
            masking.mask_dict[name][:] = (
                torch.abs(weight.data) > masking.prune_threshold
            )

    return int(total_removed)


def struct_magnitude_prune(
    masking: "Masking",
    mask: "Tensor",
    weight: "Tensor",
    name: str,
    criterion: Callable = torch.mean,
) -> "Tensor":
    """
    Prunes the weights channel-wise,
    with reduced smallest magnitude.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param mask: layer mask
    :type mask: torch.Tensor
    :param weight: layer weight
    :type weight: torch.Tensor
    :param name: layer name
    :type name: str
    :param criterion: determines reduction function.
        (reduces a kernel to a single statisitc,
        eg: mean/max/min).
    :type criterion: Callable
    :return: pruned mask
    :rtype: torch.Tensor
    """
    c_in, c_out, h, w = weight.shape

    kernel_size = h * w

    num_remove = math.ceil(
        masking.name2prune_rate[name] * masking.stats.nonzeros_dict[name] / kernel_size
    )

    if num_remove == 0.0:
        return mask

    num_zeros = masking.stats.zeros_dict[name] / kernel_size
    k = int(num_zeros + num_remove)

    reduced = criterion(
        rearrange(weight.data, "c_in c_out h w -> c_in c_out (h w)"), dim=-1
    )

    x, idx = torch.sort(torch.abs(reduced.view(-1)))

    mask.data.view(-1, h, w)[idx[:k], :, :] = 0.0
    return mask


registry = {
    "global-magnitude": global_magnitude_prune,
    "magnitude": magnitude_prune,
    "struct-magnitude-max": partial(
        struct_magnitude_prune, criterion=lambda x, **kwargs: torch.max(x, **kwargs)[0]
    ),
    "struct-magnitude-mean": partial(struct_magnitude_prune, criterion=torch.mean),
}
