"""
Comparing Tim Dettmer's and GoogleAI's implementation of the ERK sparsity distribution
"""

import logging
import numpy as np
import torch
from torch import nn


def tim_ERK(module, density, tolerance: int = 5, growth_factor: float = 0.5):
    total_params = 0
    baseline_nonzero = 0
    masks = {}
    for e, (name, weight) in enumerate(module.named_parameters()):
        # Exclude first layer
        if e == 0:
            continue
        # Exclude bias
        if "bias" in name:
            continue
        # Exclude batchnorm
        if "bn" in name:
            continue

        device = weight.device
        masks[name] = torch.zeros_like(
            weight, dtype=torch.float32, requires_grad=False
        ).to(device)

    for e, (name, weight) in enumerate(module.named_parameters()):
        if name not in masks:
            continue
        total_params += weight.numel()

    target_params = total_params * density
    current_params = 0
    epsilon = 10.0

    # searching for the right epsilon for a specific sparsity level
    while abs(current_params - target_params) > tolerance:
        new_nonzeros = 0.0
        for name, weight in module.named_parameters():
            if name not in masks:
                continue
            # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
            # we adapt the same formula for convolutional weights
            growth = max(int(epsilon * sum(weight.shape)), weight.numel())
            new_nonzeros += growth
        current_params = new_nonzeros
        if current_params > target_params:
            epsilon *= 1.0 - growth_factor
        else:
            epsilon *= 1.0 + growth_factor
        growth_factor *= 0.95

    density_dict = {}
    for name, weight in module.named_parameters():
        if name not in masks:
            continue
        growth = epsilon * sum(weight.shape)
        prob = growth / np.prod(weight.shape)
        density_dict[name] = prob
        logging.info(f"ERK {name}: {weight.shape} prob {prob}")

        device = weight.device
        masks[name] = (torch.rand(weight.shape) < prob).float().data.to(device)
        baseline_nonzero += (masks[name] != 0).sum().int().item()
    logging.info(f"Overall sparsity {baseline_nonzero/total_params}")

    return density_dict


def googleAI_ERK(module, density, erk_power_scale: float = 1.0):
    """Given the method, returns the sparsity of individual layers as a dict.
    It ensures that the non-custom layers have a total parameter count as the one
    with uniform sparsities. In other words for the layers which are not in the
    custom_sparsity_map the following equation should be satisfied.
    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    Args:
      module: 
      density: float, between 0 and 1.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.
    Returns:
      density_dict, dict of where keys() are equal to all_masks and individiual
        masks are mapped to the their densities.
    """
    # Obtain masks
    masks = {}
    total_params = 0
    for e, (name, weight) in enumerate(module.named_parameters()):
        # Exclude first layer
        if e == 0:
            continue
        # Exclude bias
        if "bias" in name:
            continue
        # Exclude batchnorm
        if "bn" in name:
            continue

        device = weight.device
        masks[name] = torch.zeros_like(
            weight, dtype=torch.float32, requires_grad=False
        ).to(device)
        total_params += weight.numel()

    # We have to enforce custom sparsities and then find the correct scaling
    # factor.

    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                    np.sum(mask.shape) / np.prod(mask.shape)
                ) ** erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    logging.info(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        logging.info(
            f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
        )
        total_nonzero += density_dict[name] * mask.numel()
    logging.info(f"Overall sparsity {total_nonzero/total_params}")
    return density_dict


if __name__ == "__main__":
    from models.wide_resnet import WideResNet

    model = WideResNet(depth=22, widen_factor=2)

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    tim_ERK(model, density=0.2)

    logging.info("========")

    googleAI_ERK(model, density=0.2)
