import math
import torch
import logging

def magnitude_prune(masking, mask, weight, name):
    """Prunes the weights with smallest magnitude.

    The pruning functions in this sparse learning library
    work by constructing a binary mask variable "mask"
    which prevents gradient flow to weights and also
    sets the weights to zero where the binary mask is 0.
    Thus 1s in the "mask" variable indicate where the sparse
    network has active weights. In this function name
    and masking can be used to access global statistics
    about the specific layer (name) and the sparse network
    as a whole.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        mask        The binary mask. 1s indicated active weights.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

    Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed

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


def global_magnitude_prune(masking):
    prune_rate = 0.0
    for name in masking.name2prune_rate:
        if name in masking.masks:
            prune_rate = masking.name2prune_rate[name]
    tokill = math.ceil(prune_rate * masking.baseline_nonzero)
    total_removed = 0
    prev_removed = 0
    while total_removed < tokill * (1.0 - masking.tolerance) or (
        total_removed > tokill * (1.0 + masking.tolerance)
    ):
        total_removed = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks:
                    continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.stats.nonzeros_dict[name] - remain

        if prev_removed == total_removed:
            break
        prev_removed = total_removed
        if total_removed > tokill * (1.0 + masking.tolerance):
            masking.prune_threshold *= 1.0 - masking.increment
            masking.increment *= 0.99
        elif total_removed < tokill * (1.0 - masking.tolerance):
            masking.prune_threshold *= 1.0 + masking.increment
            masking.increment *= 0.99

    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks:
                continue
            masking.masks[name][:] = torch.abs(weight.data) > masking.prune_threshold

    return int(total_removed)


def magnitude_and_negativity_prune(masking, mask, weight, name):
    num_remove = math.ceil(
        masking.name2prune_rate[name] * masking.stats.nonzeros_dict[name]
    )
    if num_remove == 0.0:
        return weight.data != 0.0

    num_zeros = masking.stats.zeros_dict[name]
    k = math.ceil(num_zeros + (num_remove / 2.0))

    # remove all weights which absolute value is smaller than threshold
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0

    # remove the most negative weights
    x, idx = torch.sort(weight.data.view(-1))
    mask.data.view(-1)[idx[math.ceil(num_remove / 2.0) :]] = 0.0

    return mask


def magnitude_variance_pruning(masking, mask, weight, name):
    """ Prunes weights which have high gradient variance and low magnitude.

    Intuition: Weights that are large are important but there is also a dimension
    of reliability. If a large weight makes a large correct prediction 8/10 times
    is it better than a medium weight which makes a correct prediction 10/10 times?
    To test this, we combine magnitude (importance) with reliability (variance of
    gradient).

    Good:
        Weights with large magnitude and low gradient variance are the most important.
        Weights with medium variance/magnitude are promising for improving network performance.
    Bad:
        Weights with large magnitude but high gradient variance hurt performance.
        Weights with small magnitude and low gradient variance are useless.
        Weights with small magnitude and high gradient variance cannot learn anything usefull.

    We here take the geometric mean of those both normalized distribution to find weights to prune.
    """
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp. We take the inverse of this to rank
    # low variance gradients higher.
    if "exp_avg_sq" not in masking.optimizer.state[weight]:
        print("Magnitude variance pruning requires the adam optimizer to be run!")
        raise Exception(
            "Magnitude variance pruning requires the adam optimizer to be run!"
        )
    iv_adam_sumsq = 1.0 / torch.sqrt(masking.optimizer.state[weight]["exp_avg_sq"])

    num_remove = math.ceil(
        masking.name2prune_rate[name] * masking.stats.nonzeros_dict[name]
    )

    num_zeros = masking.stats.zeros_dict[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0:
        return weight.data != 0.0

    max_var = iv_adam_sumsq[mask.bool()].max().item()
    max_magnitude = torch.abs(weight.data[mask.bool()]).max().item()
    product = (
        (iv_adam_sumsq / max_var) * torch.abs(weight.data) / max_magnitude
    ) * mask
    product[mask == 0] = 0.0

    x, idx = torch.sort(product.view(-1))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


registry = {
    "global_magnitude": global_magnitude_prune,
    "magnitude": magnitude_prune,
    "magnitude_negativity": magnitude_and_negativity_prune,
    "SET": magnitude_and_negativity_prune,
}
