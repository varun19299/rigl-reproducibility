import logging
from functools import partial
from typing import TYPE_CHECKING

from einops import repeat
import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *
from sparselearning.utils.ops import random_perm


def _remove_fc_adjust_density(masking: "Masking"):
    """
    Remove fully connected layers from masking
    (make it dense)
    and lower density of remaining layers to
    retain previous density.
    """
    n_fc = 0
    n_conv = 0

    for i, (name, module) in enumerate(masking.module.named_modules()):
        if isinstance(module, nn.Linear):
            masking.remove_weight(name)
            logging.info(
                f"Removing layer {name} of size {module.weight.numel()} parameters."
            )
            n_fc += module.weight.numel()
            continue

        if isinstance(module, nn.Conv2d):
            n_conv += module.weight.numel()

    masking.density = (masking.density * (n_conv + n_fc) - n_fc) / n_conv


def get_erdos_renyi_dist(
    masking: "Masking", is_kernel: bool = True
) -> "Dict[str, float]":
    """
    Get layer-wise densities distributed according to
    ER or ERK (erdos-renyi or erdos-renyi-kernel).

    Ensures resulting densities do not cross 1
    for any layer.

    :param masking: Masking instance
    :param is_kernel: use ERK (True), ER (False)
    :return: Layer-wise density dict
    """
    # Same as Erdos Renyi with modification for conv
    # initialization used in sparse evolutionary training
    # scales the number of non-zero weights linearly proportional
    # to the product of all dimensions, that is input*output
    # for fully connected layers, and h*w*in_c*out_c for conv
    # layers.
    _erk_power_scale = 1.0

    epsilon = 1.0
    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    _dense_layers = set()
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
        for name, mask in masking.mask_dict.items():
            n_param = np.prod(mask.shape)
            n_zeros = int(n_param * (1 - masking.density))
            n_ones = int(n_param * masking.density)

            if name in _dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones

                if is_kernel:
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                        np.sum(mask.shape) / np.prod(mask.shape)
                    ) ** _erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                else:
                    # Cin and Cout for a conv kernel
                    n_in, n_out = mask.shape[:2]
                    raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
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
                    logging.info(f"Density of layer:{mask_name} set to 1.0")
                    _dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    prob_dict = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, weight in masking.module.named_parameters():
        if name not in masking.mask_dict:
            continue

        if name in _dense_layers:
            prob = 1.0
        else:
            prob = epsilon * raw_probabilities[name]

        prob_dict[name] = prob

    return prob_dict


def erdos_renyi_init(masking: "Masking", is_kernel: bool = True, **kwargs):
    prob_dict = get_erdos_renyi_dist(masking, is_kernel)

    for name, weight in masking.module.named_parameters():
        if name not in masking.mask_dict:
            continue
        prob = prob_dict[name]
        logging.debug(f"ERK {name}: {weight.shape} prob {prob:.4f}")

        masking.mask_dict[name] = (torch.rand(weight.shape) < prob).float().data
        masking.baseline_nonzero += (masking.mask_dict[name] != 0).sum().int().item()
        masking.total_params += weight.numel()


def lottery_ticket_init(
    masking: "Masking", lottery_mask_path: "Path", shuffle: bool = False
):
    """
    Shuffle: use layer wise densities, but not exact mask
    """
    assert lottery_mask_path.is_file(), f"No .pth file at {lottery_mask_path}"
    state_dict = torch.load(lottery_mask_path, map_location="cpu")
    assert "mask" in state_dict, f"No mask found at {lottery_mask_path}"
    setattr(masking, "masks", state_dict["mask"]["masks"])

    for name, weight in masking.module.named_parameters():
        # Skip modules we arent masking
        if name not in masking.mask_dict:
            continue

        if shuffle:
            masking.mask_dict[name] = random_perm(masking.mask_dict[name])

        masking.baseline_nonzero += masking.mask_dict[name].sum().int().item()
        masking.total_params += weight.numel()

    logging.info(
        f"Loaded mask from {lottery_mask_path} with density: {masking.baseline_nonzero/masking.total_params}"
    )


def random_init(masking: "Masking", **kwargs):
    # initializes each layer with a random percentage of dense weights
    # each layer will have weight.numel()*density weights.
    # weight.numel()*density == weight.numel()*(1.0-sparsity)

    for e, (name, weight) in enumerate(masking.module.named_parameters()):
        # In random init, skip first layer
        if e == 0:
            masking.remove_weight(name)
            logging.info(
                f"Removing (first layer) {name} of size {weight.numel()} parameters."
            )
            continue

        # Skip modules we arent masking
        if name not in masking.mask_dict:
            continue

        logging.debug(
            f"Structured Random {name}: {weight.shape} prob {masking.density:.4f}"
        )

        masking.mask_dict[name] = (torch.rand(weight.shape) < masking.density).float().data
        masking.baseline_nonzero += masking.mask_dict[name].sum().int().item()
        masking.total_params += weight.numel()


def resume_init(masking: "Masking", **kwargs):
    # Initializes the mask according to the weights
    # which are currently zero-valued. This is required
    # if you want to resume a sparse model but did not
    # save the mask.
    for name, weight in masking.module.named_parameters():
        # Skip modules we arent masking
        if name not in masking.mask_dict:
            continue

        masking.mask_dict[name] = (weight != 0.0).float().data
        masking.baseline_nonzero += masking.mask_dict[name].sum().int().item()
        masking.total_params += weight.numel()
        logging.debug(
            f"{name} shape : {weight.shape} non-zero: {(weight != 0.0).sum().int().item()} density: {(weight != 0.0).sum().int().item() / weight.numel()}"
        )

    logging.info(f"Overall sparsity {masking.baseline_nonzero / masking.total_params}")


def struct_erdos_renyi_init(masking: "Masking", is_kernel: bool = True, **kwargs):
    _remove_fc_adjust_density(masking)

    prob_dict = get_erdos_renyi_dist(masking, is_kernel)

    for i, (name, weight) in enumerate(masking.module.named_parameters()):
        # Skip modules we arent masking
        if name not in masking.mask_dict:
            continue

        prob = prob_dict[name]
        logging.debug(f"Structured ERK {name}: {weight.shape} prob {prob:.4f}")

        # Allocate channel wise
        c_in, c_out, h, w = weight.shape
        A = (torch.rand(c_in, c_out, 1, 1) < prob).float()
        A = repeat(A, f"c_in c_out 1 1 -> c_in c_out {h} {w}")

        masking.mask_dict[name] = A

        masking.baseline_nonzero += masking.mask_dict[name].sum().int().item()
        masking.total_params += weight.numel()


def struct_random_init(masking: "Masking", **kwargs):
    _remove_fc_adjust_density(masking)

    for i, (name, weight) in enumerate(masking.module.named_parameters()):
        # Skip modules we arent masking
        if name not in masking.mask_dict:
            continue

        logging.debug(
            f"Structured Random {name}: {weight.shape} prob {masking.density:.4f}"
        )

        # Allocate channel wise
        c_in, c_out, h, w = weight.shape
        A = (torch.rand(c_in, c_out, 1, 1) < masking.density).float()
        A = repeat(A, f"c_in c_out 1 1 -> c_in c_out {h} {w}")

        masking.mask_dict[name] = A

        masking.baseline_nonzero += masking.mask_dict[name].sum().int().item()
        masking.total_params += weight.numel()


registry = {
    "erdos-renyi": partial(erdos_renyi_init, is_kernel=False),
    "erdos-renyi-kernel": partial(erdos_renyi_init, is_kernel=True),
    "lottery-ticket": lottery_ticket_init,
    "lottery-ticket-dist": partial(lottery_ticket_init, shuffle=True),
    "random": random_init,
    "resume": resume_init,
    "struct-erdos-renyi": partial(struct_erdos_renyi_init, is_kernel=False),
    "struct-erdos-renyi-kernel": partial(struct_erdos_renyi_init, is_kernel=True),
    "struct-random": struct_random_init,
}
