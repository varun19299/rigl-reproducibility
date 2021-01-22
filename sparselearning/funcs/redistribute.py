"""
Implements Redistribution function.

Modifies layer-wise sparsity during mask update.

Functions have access to the masking object
enabling greater flexibility in designing
custom redistribution modes.

Masking class implements the output redistribution
in a valid manner, ensuring no weight exceeds its capacity.

Signature:
<func>(masking, name, weight, mask)
"""
import torch


def momentum_redistribution(masking, name, weight, mask)-> float:
    """
    Calculates momentum redistribution statistics.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param weight: layer weight
    :type weight: torch.Tensor
    :param mask: layer mask
    :type mask: torch.Tensor
    :return: Layer Statistic---unnormalized layer statistics
            for the layer. Normalizing across layers gives
            the density distribution.
    :rtype: float
    """
    momentum = masking.get_momentum_for_weight(weight)

    mean_magnitude = torch.abs(momentum[mask.bool()]).mean().item()
    return mean_magnitude


def grad_redistribution(masking, name, weight, mask):
    """
    Calculates gradient redistribution statistics.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param weight: layer weight
    :type weight: torch.Tensor
    :param mask: layer mask
    :type mask: torch.Tensor
    :return: Layer Statistic---unnormalized layer statistics
            for the layer. Normalizing across layers gives
            the density distribution.
    :rtype: float
    """
    grad = weight.grad
    mean_grad = torch.abs(grad[mask.bool()]).mean().item()
    return mean_grad


def nonzero_redistribution(masking, name, weight, mask):
    """
    Calculates non-zero redistribution statistics.
    Ideally, this just preserves the older distribution,
    upto numerical error.
    In practice, we prefer to skip redistribution if
    non-zero is chosen.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param name: layer name
    :type name: str
    :param weight: layer weight
    :type weight: torch.Tensor
    :param mask: layer mask
    :type mask: torch.Tensor
    :return: Layer Statistic---unnormalized layer statistics
            for the layer. Normalizing across layers gives
            the density distribution.
    :rtype: float
    """
    nonzero = (weight != 0.0).sum().item()
    return nonzero


registry = {
    "grad": grad_redistribution,
    "momentum": momentum_redistribution,
    "nonzero": nonzero_redistribution,
    "none": nonzero_redistribution,
}
