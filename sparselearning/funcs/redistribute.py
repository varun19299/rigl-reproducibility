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


def momentum_redistribution(masking, name, weight, mask):
    """
    Calculates momentum redistribution statistics.

    :param masking:
    :param name: layer name
    :param weight: layer weight
    :param mask: layer mask
    :return: Layer Statistic---unnormalized layer statistics
        for the layer. Normalizing across layers gives
        the density distribution.
    """
    momentum = masking.get_momentum_for_weight(weight)

    mean_magnitude = torch.abs(momentum[mask.bool()]).mean().item()
    return mean_magnitude


def grad_redistribution(masking, name, weight, mask):
    """Calculates gradient redistribution statistics.

    :param masking:
    :param name: layer name
    :param weight: layer weight
    :param mask: layer mask
    :return: Layer Statistic---unnormalized layer statistics
        for the layer. Normalizing across layers gives
        the density distribution.
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

    :param masking:
    :param name: layer name
    :param weight: layer weight
    :param mask: layer mask
    :return: Layer Statistic---unnormalized layer statistics
        for the layer. Normalizing across layers gives
        the density distribution.
    """
    nonzero = (weight != 0.0).sum().item()
    return nonzero


registry = {
    "grad": grad_redistribution,
    "momentum": momentum_redistribution,
    "nonzero": nonzero_redistribution,
    "none": nonzero_redistribution,
}
