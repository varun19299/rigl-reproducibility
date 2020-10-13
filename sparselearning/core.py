from dataclasses import dataclass, field

import logging
import math
import numpy as np

# Sparse learning funcs
from sparselearning.funcs.decay import CosineDecay, LinearDecay
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

    prune_rate: float = 0.5
    prune_mode: str = "magnitude"
    growth_mode: str = "momentum"
    redistribution_mode: str = "momentum"
    # Print param counts etc after each mask update
    verbose: bool = False

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

    def __post_init__(self):
        self.masks = {}
        self.modules = []
        self.names = []

        self.adjusted_growth = 0
        self.adjustments = []

        # stats
        self.name2prune_rate = {}

        self.steps = 0

        self.stats = LayerStats()

    """
    Code flow:
    
    1. add_module():
        * init()
        * optionally remove_...()
        * apply_mask()
        
    2. step():
        * apply_mask()
        (why? because gradients will also modify non-zero weights)
        * step prune_rate_decay()
            
    3. update_connections():
        * truncate_weights()
            * 
            * 
        * print_nonzero_counts() [if verbose]
        
    Unclear: calc_growth_redistribution    
    """

    def add_module(self, module):
        """
        Store dict of parameters to mask
        """
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            device = tensor.device
            self.masks[name] = torch.zeros_like(
                tensor, dtype=torch.float32, requires_grad=False
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
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue

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
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

    def calc_growth_redistribution(self):
        name2regrowth = {}
        # prune_rate_ll = torch.tensor(list(self.name2prune_rate.values()))
        # zero_ll = torch.tensor(list(self.stats.zeros_dict.values()))
        # nonzero_ll = torch.tensor(list(self.stats.nonzeros_dict.values()))
        # remove_ll = torch.ceil(prune_rate_ll * nonzero_ll)
        #
        # # Upper bound on total new connections
        # max_regrowth_ll = zero_ll + remove_ll
        #
        # # Estimated layer wise via the variance
        # # If variance is 1.0,
        # variance_ll = torch.tensor(list(self.stats.variance_dict.values()))
        # regrowth_ll = torch.ceil(
        #     variance_ll * (self.stats.total_removed + self.adjusted_growth)
        # )
        # residual_ll = regrowth_ll - 0.99 * max_regrowth_ll
        # regrowth_ll[regrowth_ll > 0.99 * max_regrowth_ll] = 0.99 * max_regrowth_ll
        # mean_residual = residual_ll.mean()
        #
        # assert mean_residual

        mean_residual = 0

        for i in range(1001):
            residual = 0
            for name in self.stats.variance_dict:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate * self.stats.nonzeros_dict[name])
                num_zero = self.stats.zeros_dict[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(
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

            # Non-positive residual
            if residual <= 0:
                break

        if i == 1000:
            logging.info(
                f"Error resolving the residual! Layers are too full! Residual left over: {residual}"
            )

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                if self.prune_mode == "global_magnitude":
                    expected_removed = (
                        self.baseline_nonzero * self.name2prune_rate[name]
                    )
                    if expected_removed == 0.0:
                        name2regrowth[name] = 0.0
                    else:
                        expected_vs_actual = self.stats.total_removed / expected_removed
                        name2regrowth[name] = math.floor(
                            expected_vs_actual * name2regrowth[name]
                        )

        return name2regrowth

    def init(self, mode="random", density=0.05):

        # Number of params originally non-zero
        # Total params * inital density
        self.baseline_nonzero = 0

        if self.sparse_init == "random":
            # initializes each layer with a random percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)

            for module in self.modules:
                for e, (name, weight) in enumerate(module.named_parameters()):
                    # In random init, skip first layer
                    if e == 0:
                        self.remove_weight(name)
                        continue

                    # Skip modules we arent masking
                    if name not in self.masks:
                        continue

                    device = self.masks[name].device
                    self.masks[name] = (
                        (torch.rand(weight.shape) < self.density)
                        .float()
                        .data.to(device)
                    )
                    self.baseline_nonzero += weight.numel() * self.density
            self.apply_mask()

        elif self.sparse_init == "resume":
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            for module in self.modules:
                for name, weight in module.named_parameters():
                    # Skip modules we arent masking
                    if name not in self.masks:
                        continue

                    logging.info((weight != 0.0).sum().item())
                    device = weight.device
                    self.masks[name] = (weight != 0.0).float().data.to(device)
                    self.baseline_nonzero += weight.numel() * self.density
            self.apply_mask()

        elif self.sparse_init == "erdos_renyi":
            # Same as Erdos Renyi with modification for conv
            # initialization used in sparse evolutionary training
            # scales the number of non-zero weights linearly proportional
            # to the product of all dimensions, that is input*output
            # for fully connected layers, and h*w*in_c*out_c for conv
            # layers.

            total_params = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel() * self.density

            target_params = total_params * self.density
            tolerance = 5
            current_params = 0
            # TODO: is the below needed
            # Can we do this more elegantly?
            # new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5

            # searching for the right epsilon for a specific sparsity level
            while abs(current_params - target_params) < tolerance:
                new_nonzeros = 0.0
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth = epsilon * sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                growth = epsilon * sum(weight.shape)
                prob = growth / np.prod(weight.shape)

                device = weight.device
                self.masks[name] = (
                    (torch.rand(weight.shape) < prob).float().data.to(device)
                )
            self.apply_mask()

        self.print_nonzero_counts()

        total_size = 0
        for name, module in self.modules[0].named_modules():
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
            f"Total parameters under sparsity level of {self.density}: {self.density * total_size}"
        )

    def gather_statistics(self):
        name2variance = {}
        name2nonzeros = {}
        name2zeros = {}
        name2removed = {}

        total_variance = 0.0
        total_removed = 0
        total_nonzero = 0
        total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]

                # redistribution
                name2variance[name] = self.redistribution_func(self, name, weight, mask)

                if not np.isnan(name2variance[name]):
                    total_variance += name2variance[name]
                name2nonzeros[name] = mask.sum().item()
                name2zeros[name] = mask.numel() - name2nonzeros[name]

                sparsity = name2zeros[name] / float(self.masks[name].numel())
                total_nonzero += name2nonzeros[name]
                total_zero += name2zeros[name]

        for name in name2variance:
            if total_variance != 0.0:
                name2variance[name] /= total_variance
            else:
                logging.info("Total variance was zero!")
                logging.info(self.growth_func)
                logging.info(self.prune_func)
                logging.info(self.redistribution_func)
                logging.info(name2variance)

        self.stats = LayerStats(
            variance_dict=name2variance,
            nonzeros_dict=name2nonzeros,
            zeros_dict=name2zeros,
            removed_dict=name2removed,
            total_variance=total_variance,
            total_nonzero=total_nonzero,
            total_zero=total_zero,
            total_removed=total_removed,
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

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()

                if name in self.stats.variance_dict:
                    log_str = (
                        f"{name}: {self.stats.nonzeros_dict[name]}->{num_nonzeros}, "
                        f"density: {num_nonzeros / float(mask.numel()):.3f}, "
                        f"proportion: {self.stats.variance_dict[name]:.4f}"
                    )

                else:
                    log_str = f"{name}: {num_nonzeros}"
                logging.info(log_str)

        logging.info(f"Prune rate: {self.prune_rate}.")

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
            if self.verbose:
                logging.info(
                    f"Removing {name} of size {self.masks[name].shape} = {self.masks[name].numel()} parameters."
                )
            self.masks.pop(name)
        elif name + ".weight" in self.masks:
            if self.verbose:
                logging.info(
                    f"Removing {name} of size {self.masks[name + '.weight'].shape} = {self.masks[name + '.weight'].numel()} parameters."
                )
            self.masks.pop(name + ".weight")
        else:
            if self.verbose:
                logging.info(f"ERROR {name} not found.")

    def remove_weight_partial_name(self, partial_name: str):
        """
        Remove module by partial name (eg: conv).
        """
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    logging.info(
                        f"Removing {name} of size {self.masks[name].shape} with {self.masks[name].numel()} parameters."
                    )
                removed.add(name)
                self.masks.pop(name)

        # Update names
        self.names = [name for name in self.names if name not in removed]

        if self.verbose:
            logging.info(f"Removed {len(removed)} layers.")

    def remove_type(self, nn_type):
        """
        Remove module by type (eg: nn.Linear, nn.Conv2d, etc.)
        """
        for module in self.modules:
            for name, module in module.named_modules():
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
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)

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
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue
                    mask = self.masks[name]

                    # prune
                    new_mask = self.prune_func(self, mask, weight, name)
                    removed = self.stats.nonzeros_dict[name] - new_mask.sum().item()
                    self.stats.total_removed += removed
                    self.stats.removed_dict[name] = removed
                    self.masks[name] = new_mask

        name2regrowth = self.calc_growth_redistribution()
        if self.global_growth:
            total_nonzero_new = self.growth_func(
                self, self.stats.total_removed + self.adjusted_growth
            )
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue
                    new_mask = self.masks[name].data.bool()

                    # growth
                    new_mask = self.growth_func(
                        self, name, new_mask, math.floor(name2regrowth[name]), weight
                    )
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (prune-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = (
            0.25 * self.adjusted_growth
            + (0.75 * (self.baseline_nonzero - total_nonzero_new))
            + np.mean(self.adjustments)
        )
        if self.stats.total_nonzero > 0 and self.verbose:
            logging.info(
                f"Nonzero before/after: {self.stats.total_nonzero}/{total_nonzero_new}. "
                f"Growth adjustment: {self.adjusted_growth:.2f}."
            )

    def update_connections(self):
        self.truncate_weights()
        if self.verbose:
            self.print_nonzero_counts()
