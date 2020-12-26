import math

import torch


def momentum_growth(masking, name, new_mask, total_regrowth, weight):
    """Grows weights in places where the momentum is largest.

    Growth function in the sparse learning library work by
    changing 0s to 1s in a binary mask which will enable
    gradient flow. Weights default value are 0 and it can
    be changed in this function. The number of parameters
    to be regrown is determined by the total_regrowth
    parameter. The masking object in conjunction with the name
    of the layer enables the access to further statistics
    and objects that allow more flexibility to implement
    custom growth functions.

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


def abs_grad_growth(masking, name, new_mask, total_regrowth, weight):
    """Grows weights in places where the abs(grad) is largest. (among present zero'ed weights)"""
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


def random_growth(masking, name, new_mask, total_regrowth, weight):
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


def momentum_neuron_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)

    M = torch.abs(grad)
    if len(M.shape) == 2:
        sum_dim = [1]
    elif len(M.shape) == 4:
        sum_dim = [1, 2, 3]

    v = M.mean(sum_dim).data
    v /= v.sum()

    slots_per_neuron = (new_mask == 0).sum(sum_dim)

    M = M * (new_mask == 0).float()
    for i, fraction in enumerate(v):
        neuron_regrowth = math.floor(fraction.item() * total_regrowth)
        available = slots_per_neuron[i].item()

        y, idx = torch.sort(M[i].flatten())
        if neuron_regrowth > available:
            neuron_regrowth = available
        # TODO: Work into more stable growth method
        threshold = y[-(neuron_regrowth)].item()
        if threshold == 0.0:
            continue
        if neuron_regrowth < 10:
            continue
        new_mask[i] = new_mask[i] | (M[i] > threshold)

    return new_mask


def no_growth(masking, name, new_mask, total_regrowth, weight):
    """
    No growth
    """
    return new_mask


def global_momentum_growth(masking, total_regrowth):
    togrow = total_regrowth
    total_grown = 0
    last_grown = 0
    while total_grown < togrow * (1.0 - masking.tolerance) or (
        total_grown > togrow * (1.0 + masking.tolerance)
    ):
        total_grown = 0
        total_possible = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks:
                    continue

                new_mask = masking.masks[name]
                grad = masking.get_momentum_for_weight(weight)
                grad = grad * (new_mask == 0).float()
                possible = (grad != 0.0).sum().item()
                total_possible += possible
                grown = (torch.abs(grad.data) > masking.growth_threshold).sum().item()
                total_grown += grown
        if total_grown == last_grown:
            break
        last_grown = total_grown

        if total_grown > togrow * (1.0 + masking.tolerance):
            masking.growth_threshold *= 1.02
            # masking.growth_increment *= 0.95
        elif total_grown < togrow * (1.0 - masking.tolerance):
            masking.growth_threshold *= 0.98
            # masking.growth_increment *= 0.95

    total_new_nonzeros = 0
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks:
                continue

            new_mask = masking.masks[name]
            grad = masking.get_momentum_for_weight(weight)
            grad = grad * (new_mask == 0).float()
            masking.masks[name][:] = (
                new_mask.bool() | (torch.abs(grad.data) > masking.growth_threshold)
            ).float()
            total_new_nonzeros += new_mask.sum().item()
    return total_new_nonzeros


def block_abs_grad_growth(masking, name, new_mask, total_regrowth, weight, criterion):
    # If dense, skip
    n = (new_mask == 0).sum().item()
    if n == 0:
        return new_mask

    grad = weight.grad
    if grad.dtype == torch.float16:
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()

    kernel_size = grad.shape[-1] ** 2

    reduced = criterion(grad.view(*grad.shape[:2], -1), axis=-1)

    y, idx = torch.sort(torch.abs(reduced).flatten(), descending=True)

    new_mask.data.view(-1, *weight.shape[-2:])[idx[: int(total_regrowth)], :, :] = 1.0

    # init new weights to 0
    weight.data.view(-1, *weight.shape[-2:])[idx[: int(total_regrowth)], :, :] = 0.0

    return new_mask


registry = {
    "absolute-gradient": abs_grad_growth,
    "global-momentum-growth": global_momentum_growth,
    "momentum": momentum_growth,
    "momentum-neuron": momentum_neuron_growth,
    "none": no_growth,
    "random": random_growth,
    "block-absolute-gradient": block_abs_grad_growth,
}
