from dataclasses import dataclass, field

import logging
import math
import numpy as np

# Sparse learning funcs
from sparselearning.funcs.grow import registry as grow_registry
from sparselearning.funcs.prune import registry as prune_registry
from sparselearning.funcs.redistribute import registry as redistribute_registry

import torch
import torch.nn as nn
import torch.optim as optim

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


@dataclass
class LayerStats(object):
    # maps layer names to variances
    variance_dict: "Dict[str, float]" = field(default_factory=dict)
    # maps layer names to no of zeroed weights
    zeros_dict: "Dict[str, int]" = field(default_factory=dict)
    # maps layer names to no of non-zero weights
    nonzeros_dict: "Dict[str, int]" = field(default_factory=dict)

    # maps layer names to removed weights
    # (w.r.t to base initialization)
    removed_dict: "Dict[str, int]" = field(default_factory=dict)

    # Same stats across network
    total_variance: float = 0
    total_zero: int = 0
    total_nonzero: int = 0
    total_removed: int = 0


@dataclass
class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)

    Removing layers: Layers can be removed individually, by type, or by partial
    match of their name.
      - `mask.remove_weight(name)` requires an exact name of
    a parameter.
      - `mask.remove_weight_partial_name(partial_name=name)` removes all
        parameters that contain the partial name. For example 'conv' would remove all
        layers with 'conv' in their name.
      - `mask.remove_type(type)` removes all layers of a certain type. For example,
        mask.remove_type(torch.nn.BatchNorm2d) removes all 2D batch norm layers.
    """

    optimizer: "optim"
    prune_rate_decay: "Decay"

    density: float  # Sparsity = 1 - density
    sparse_init: str = "random"  # or erdos_renyi

    prune_mode: str = "magnitude"
    growth_mode: str = "momentum"
    redistribution_mode: str = "momentum"

    # Apply growth or prune func to entire module
    # Instead of layer wise
    global_growth: bool = False
    global_prune: bool = False
    # global growth/prune state
    prune_threshold: float = 0.001
    growth_threshold: float = 0.001
    growth_increment: float = 0.2
    increment: float = 0.2
    tolerance: float = 0.02

    # mask & module
    masks: "Dict[str, Tensor]" = field(default_factory=dict)
    module: "nn.Module" = None  # Pytorch module

    # Growth adjustment
    adjusted_growth: float = 0
    adjustments: "List[float]" = field(default_factory=list)

    # stats
    steps: int = 0
    name2prune_rate: "Dict[str, float]" = field(default_factory=dict)
    stats: "LayerStats" = LayerStats()

    """
    Code flow:
    
    1. add_module():
        * init()
        * optionally remove_...()
        * apply_mask()
        
    2. step():
        * apply_mask()
        * prune_rate_decay()
            
    3. update_connections():
        * truncate_weights()
            * prune
            * redistribute (optional)
            * grow
        * print_nonzero_counts() [if verbose]
    """

    def add_module(self, module):
        """
        Store dict of parameters to mask
        """
        self.module = module
        for name, weight in self.module.named_parameters():
            device = weight.device
            self.masks[name] = torch.zeros_like(
                weight, dtype=torch.float32, requires_grad=False
            ).to(device)

        # Remove bias, batchnorms
        logging.info("Removing biases...")
        self.remove_weight_partial_name("bias")
        logging.info("Removing 2D batch norms...")
        self.remove_type(nn.BatchNorm2d)
        logging.info("Removing 1D batch norms...")
        self.remove_type(nn.BatchNorm1d)

        # Call init
        self.init()

    def adjust_prune_rate(self):
        """
        Modify prune rate for layers with low sparsity
        """
        for name in self.names:
            self.name2prune_rate[name] = self.prune_rate

            sparsity = self.stats.zeros_dict[name] / self.masks[name].numel()
            if sparsity < 0.2:
                # determine if matrix is relatively dense but still growing
                expected_variance = 1.0 / len(self.stats.variance_dict.keys())
                actual_variance = self.stats.variance_dict[name]
                expected_vs_actual = expected_variance / actual_variance

                # if weights arent steady yet, i.e., can change significantly
                if expected_vs_actual < 1.0:
                    # growing
                    self.name2prune_rate[name] = min(
                        sparsity, self.name2prune_rate[name]
                    )

    def apply_mask(self):
        """
        Applies boolean mask to modules
        """
        for name, weight in self.module.named_parameters():
            if name in self.masks:
                weight.data = weight.data * self.masks[name]

    def calc_growth_redistribution(self):
        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.stats.variance_dict:
                # prune_rate = self.name2prune_rate[name]
                # num_remove = math.ceil(prune_rate * self.stats.nonzeros_dict[name])
                num_remove = self.stats.removed_dict[name]
                # num_nonzero = self.stats.nonzeros_dict[name]
                num_zero = self.stats.zeros_dict[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = round(
                        self.stats.variance_dict[name]
                        * (self.stats.total_removed + self.adjusted_growth)
                    )
                regrowth += mean_residual

                if regrowth > 0.99 * max_regrowth:
                    name2regrowth[name] = 0.99 * max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0:
                mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print(
                f"Error resolving the residual! Layers are too full! Residual left over: {residual}"
            )

        if self.prune_mode == "global_magnitude":
            for name in self.masks:
                expected_removed = self.baseline_nonzero * self.name2prune_rate[name]
                if expected_removed == 0.0:
                    name2regrowth[name] = 0.0
                else:
                    expected_vs_actual = self.stats.total_removed / expected_removed
                    name2regrowth[name] = math.floor(
                        expected_vs_actual * name2regrowth[name]
                    )

        return name2regrowth

    def init(self):

        # Number of params originally non-zero
        # Total params * inital density
        self.baseline_nonzero = 0

        if self.sparse_init == "random":
            # initializes each layer with a random percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)

            for e, (name, weight) in enumerate(self.module.named_parameters()):
                # In random init, skip first layer
                if e == 0:
                    self.remove_weight(name)
                    logging.info(
                        f"Removing (first layer) {name} of size {weight.numel()} parameters."
                    )
                    continue

                # Skip modules we arent masking
                if name not in self.masks:
                    continue

                device = self.masks[name].device
                self.masks[name] = (
                    (torch.rand(weight.shape) < self.density).float().data.to(device)
                )
                # self.baseline_nonzero += weight.numel() * self.density
                self.baseline_nonzero += self.masks[name].sum().int().item()

        elif self.sparse_init == "resume":
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            for name, weight in self.module.named_parameters():
                # Skip modules we arent masking
                if name not in self.masks:
                    continue

                logging.info((weight != 0.0).sum().item())
                device = weight.device
                self.masks[name] = (weight != 0.0).float().data.to(device)
                # self.baseline_nonzero += weight.numel() * self.density
                self.baseline_nonzero += self.masks[name].sum().int().item()

        elif self.sparse_init == "erdos_renyi":
            # Same as Erdos Renyi with modification for conv
            # initialization used in sparse evolutionary training
            # scales the number of non-zero weights linearly proportional
            # to the product of all dimensions, that is input*output
            # for fully connected layers, and h*w*in_c*out_c for conv
            # layers.

            total_params = 0
            for e, (name, weight) in enumerate(self.module.named_parameters()):
                # Skip first layer
                if e == 0:
                    self.remove_weight(name)
                    logging.info(
                        f"Removing (first layer) {name} of size {weight.numel()} parameters."
                    )
                    continue

                if name not in self.masks:
                    continue
                total_params += weight.numel()
                # self.baseline_nonzero += weight.numel() * self.density

            _erk_power_scale = 1.0

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
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = int(n_param * (1 - self.density))
                    n_ones = int(n_param * self.density)

                    if name in _dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                            np.sum(mask.shape) / np.prod(mask.shape)
                        ) ** _erk_power_scale
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
                            logging.info(
                                f"Density of layer:{mask_name} set to 1.0"
                            )
                            _dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, weight in self.module.named_parameters():
                if name not in self.masks:
                    continue

                if name in _dense_layers:
                    prob = 1.0
                else:
                    prob = epsilon * raw_probabilities[name]
                logging.debug(f"ERK {name}: {weight.shape} prob {prob:.4f}")

                device = weight.device
                self.masks[name] = (
                    (torch.rand(weight.shape) < prob).float().data.to(device)
                )
                self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        self.apply_mask()

        self.print_nonzero_counts()

        total_size = 0
        for name, module in self.module.named_modules():
            if hasattr(module, "weight"):
                total_size += module.weight.numel()
            if hasattr(module, "bias"):
                if module.bias is not None:
                    total_size += module.bias.numel()
        logging.info(f"Total Model parameters: {total_size}.")

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        logging.info(f"Total parameters after removed layers: {total_size}.")
        logging.info(
            f"Total parameters under sparsity level of {self.density}: {self.baseline_nonzero}"
        )
        logging.info(
            f"Achieved sparsity at init (w/o BN, bias): {self.baseline_nonzero/total_params:.4f}"
        )

    def gather_statistics(self):
        variance_dict = {}
        nonzeros_dict = {}
        zeros_dict = {}

        total_variance = 0.0
        total_nonzero = 0
        total_zero = 0
        for name, weight in self.module.named_parameters():
            if name not in self.masks:
                continue
            mask = self.masks[name]

            # redistribution
            variance_dict[name] = self.redistribution_func(self, name, weight, mask)

            if not np.isnan(variance_dict[name]):
                total_variance += variance_dict[name]
            nonzeros_dict[name] = (mask != 0).sum().int().item()
            zeros_dict[name] = (mask == 0).sum().int().item()

            total_nonzero += nonzeros_dict[name]
            total_zero += zeros_dict[name]

        assert total_variance, "Total variance is zero!"
        for name in variance_dict:
            variance_dict[name] /= total_variance

        self.stats = LayerStats(
            variance_dict=variance_dict,
            nonzeros_dict=nonzeros_dict,
            zeros_dict=zeros_dict,
            total_variance=total_variance,
            total_nonzero=total_nonzero,
            total_zero=total_zero,
        )

    @property
    def growth_func(self):
        assert (
            self.growth_mode in grow_registry.keys()
        ), f"Available growth modes: {','.join(grow_registry.keys())}"
        if "global" in self.growth_mode:
            self.global_growth = True

        return grow_registry[self.growth_mode]

    def get_momentum_for_weight(self, weight):
        """
        Return momentum from optimizer (SGD or Adam)
        """
        # Adam
        if "exp_avg" in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]["exp_avg"]
            adam_m2 = self.optimizer.state[weight]["exp_avg_sq"]
            momentum = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        # SGD
        elif "momentum_buffer" in self.optimizer.state[weight]:
            momentum = self.optimizer.state[weight]["momentum_buffer"]

        return momentum

    @property
    def names(self) -> "List[str]":
        """
        Names of each masked layer
        """
        return list(self.masks.keys())

    def print_nonzero_counts(self):
        for name, mask in self.masks.items():
            num_nonzeros = (mask != 0).sum().item()

            if name in self.stats.variance_dict:
                log_str = (
                    f"{name}: {self.stats.nonzeros_dict[name]}->{num_nonzeros}, "
                    f"density: {num_nonzeros / float(mask.numel()):.3f}, "
                    f"proportion: {self.stats.variance_dict[name]:.4f}"
                )

            else:
                log_str = f"{name}: {num_nonzeros}"
            logging.debug(log_str)

        logging.debug(f"Prune rate: {self.prune_rate}.")

    @property
    def prune_func(self):
        """
        Calls prune func from the  registry.

        We use @property, so that it is always
        synced with prune_mode
        """
        assert (
            self.prune_mode in prune_registry.keys()
        ), f"Available prune modes: {','.join(prune_registry.keys())}"
        if "global" in self.prune_mode:
            self.global_prune = True
        return prune_registry[self.prune_mode]

    @property
    def prune_rate(self) -> float:
        """
        Get prune rate from the decay object
        """
        return self.prune_rate_decay.get_dr()

    @property
    def redistribution_func(self):
        """
        Calls redistribution func from the  registry.

        We use @property, so that it is always
        synced with redistribution_mode
        """
        assert (
            self.redistribution_mode in redistribute_registry.keys()
        ), f"Available redistribute modes: {','.join(redistribute_registry.keys())}"
        return redistribute_registry[self.redistribution_mode]

    def remove_weight(self, name):
        """
        Remove weight by complete name
        """
        if name in self.masks:
            logging.debug(
                f"Removing {name} of size {self.masks[name].shape} = {self.masks[name].numel()} parameters."
            )
            self.masks.pop(name)
        elif name + ".weight" in self.masks:
            logging.debug(
                f"Removing {name} of size {self.masks[name + '.weight'].shape} = {self.masks[name + '.weight'].numel()} parameters."
            )
            self.masks.pop(name + ".weight")
        else:
            logging.error(f"ERROR {name} not found.")

    def remove_weight_partial_name(self, partial_name: str):
        """
        Remove module by partial name (eg: conv).
        """
        _removed = 0
        for name in list(self.masks.keys()):
            if partial_name in name:
                logging.debug(
                    f"Removing {name} of size {self.masks[name].shape} with {self.masks[name].numel()} parameters."
                )
                _removed += 1
                self.masks.pop(name)

        logging.debug(f"Removed {_removed} layers.")

    def remove_type(self, nn_type):
        """
        Remove module by type (eg: nn.Linear, nn.Conv2d, etc.)
        """
        for name, module in self.module.named_modules():
            if isinstance(module, nn_type):
                self.remove_weight(name)

    def step(self):
        """
        Performs a masking step
        """
        self.optimizer.step()
        self.apply_mask()

        # Get updated prune rate
        self.prune_rate_decay.step()

        self.steps += 1

    def truncate_weights(self):
        """
        Perform grow / prune / redistribution step
        """
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        if self.global_prune:
            self.stats.total_removed = self.prune_func(self)
        else:
            for name, weight in self.module.named_parameters():
                # Skip modules we arent masking
                if name not in self.masks:
                    continue

                mask = self.masks[name]

                # prune
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.stats.nonzeros_dict[name] - int(new_mask.sum().item())
                self.stats.total_removed += removed
                self.stats.removed_dict[name] = removed
                self.masks[name] = new_mask

        if self.global_growth:
            total_nonzero_new = self.growth_func(
                self, self.stats.total_removed + self.adjusted_growth
            )
        else:
            if self.redistribution_mode not in ["nonzero", "none"]:
                name2regrowth = self.calc_growth_redistribution()

            for name, weight in self.module.named_parameters():
                # Skip modules we arent masking
                if name not in self.masks:
                    continue

                new_mask = self.masks[name].data.bool()

                # growth
                if self.redistribution_mode not in ["nonzero", "none"]:
                    num_growth = name2regrowth[name]
                else:
                    num_growth = self.stats.removed_dict[name]

                new_mask = self.growth_func(self, name, new_mask, num_growth, weight)
                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                total_nonzero_new += new_nonzero

        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (prune-growth) residuals to adjust future growth
        self.adjustments.append(
            self.baseline_nonzero - total_nonzero_new
        )  # will be zero if deterministic
        self.adjusted_growth = (
            0.25 * self.adjusted_growth
            + (0.75 * self.adjustments[-1])
            + np.mean(self.adjustments)
        )
        if self.stats.total_nonzero > 0:
            logging.debug(
                f"Nonzero before/after: {self.stats.total_nonzero}/{total_nonzero_new}. "
                f"Growth adjustment: {self.adjusted_growth:.2f}."
            )

    def update_connections(self):
        self.truncate_weights()
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            # debug logged
            self.print_nonzero_counts()
