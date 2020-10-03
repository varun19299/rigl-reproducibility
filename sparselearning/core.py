from dataclasses import dataclass

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
    prune_every_k_steps: int = 0

    def __post_init__(self):
        self.masks = {}
        self.modules = []
        self.names = []

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None

    """
    Code flow:
    
    1. add_module():
        * init()
        * optionally remove_...()
        * apply_mask()
        
    2. step():
        * apply_mask()
        * step prune_rate_decay()
        * if prune_every_k_steps:
            * truncate_weights()
            * print_nonzero_counts() [if verbose]
    
    2.b. truncate_weights()
        * 
            
    3. at_end_of_epoch():
        * truncate_weights()
        * print_nonzero_counts() [if verbose]
    """

    def add_module(self, module, density, sparse_init="constant"):
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
        self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        logging.info("Removing 1D batch norms...")
        self.remove_type(nn.BatchNorm1d, verbose=self.verbose)

        # Call init
        self.init(mode=sparse_init, density=density)

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue

                # TODO: removing as redundant
                # Once we reproduce SNFS, delete em!
                # if name not in self.name2prune_rate:
                #     self.name2prune_rate[name] = self.prune_rate
                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name] / float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0 / len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance / actual_variance

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

    def at_end_of_epoch(self):
        self.truncate_weights()
        if self.verbose:
            self.print_nonzero_counts()

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0 / len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate * self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(
                        self.name2variance[name]
                        * (self.total_removed + self.adjusted_growth)
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
            logging.info(
                "Error resolving the residual! Layers are too full! Residual left over: {0}".format(
                    residual
                )
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
                        expected_vs_actual = self.total_removed / expected_removed
                        name2regrowth[name] = math.floor(
                            expected_vs_actual * name2regrowth[name]
                        )

        return name2regrowth

    def init(self, mode="constant", density=0.05):
        self.sparsity = density
        self.baseline_nonzero = 0

        if mode == "constant":
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    # Skip modules we arent masking
                    if name not in self.masks:
                        continue

                    device = self.masks[name].device
                    self.masks[name][:] = (
                        (torch.rand(weight.shape) < density).float().data.to(device)
                    )
                    self.baseline_nonzero += weight.numel() * density
            self.apply_mask()

        elif mode == "resume":
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

                    self.masks[name][:] = (weight != 0.0).float().data.cuda()
                    self.baseline_nonzero += weight.numel() * density
            self.apply_mask()

        elif mode in ["linear", "ER"]:
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
                    self.baseline_nonzero += weight.numel() * density

            target_params = total_params * density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not (
                (current_params + tolerance > target_params)
                and (current_params - tolerance < target_params)
            ):
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
                self.masks[name][:] = (
                    (torch.rand(weight.shape) < prob).float().data.cuda()
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
            f"Total parameters under sparsity level of {density}: {density * total_size}"
        )

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}
        self.name2removed = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]

                # redistribution
                self.name2variance[name] = self.redistribution_func(
                    self, name, weight, mask
                )

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name] / float(self.masks[name].numel())
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

        for name in self.name2variance:
            if self.total_variance != 0.0:
                self.name2variance[name] /= self.total_variance
            else:
                logging.info("Total variance was zero!")
                logging.info(self.growth_func)
                logging.info(self.prune_func)
                logging.info(self.redistribution_func)
                logging.info(self.name2variance)

    @property
    def growth_func(self):
        assert (
            self.growth_mode in grow_registry.keys()
        ), f"Available growth modes: {','.join(grow_registry.keys())}"
        if "global" in self.growth_mode:
            self.global_growth = True

        return grow_registry[self.growth_mode]

    def get_momentum_for_weight(self, weight):
        if "exp_avg" in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]["exp_avg"]
            adam_m2 = self.optimizer.state[weight]["exp_avg_sq"]
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif "momentum_buffer" in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]["momentum_buffer"]

        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = "{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}".format(
                        name,
                        self.name2nonzeros[name],
                        num_nonzeros,
                        num_nonzeros / float(mask.numel()),
                        self.name2variance[name],
                    )
                    logging.info(val)
                else:
                    logging.info(name, num_nonzeros)

        logging.info("Prune rate: {0}\n".format(self.prune_rate))

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
        assert (
            self.redistribution_mode in redistribute_registry.keys()
        ), f"Available redistribute modes: {','.join(redistribute_registry.keys())}"
        return redistribute_registry[self.redistribution_mode]

    def remove_weight(self, name):
        """
        Remove weight by complete name
        """
        if name in self.masks:
            logging.info(
                f"Removing {name} of size {self.masks[name].shape} = {self.masks[name].numel()} parameters."
            )
            self.masks.pop(name)
        elif name + ".weight" in self.masks:
            logging.info(
                f"Removing {name} of size {self.masks[name + '.weight'].shape} = {self.masks[name + '.weight'].numel()} parameters."
            )
            self.masks.pop(name + ".weight")
        else:
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

        if self.prune_every_k_steps and (self.steps % self.prune_every_k_steps == 0):
            self.truncate_weights()
            if self.verbose:
                self.print_nonzero_counts()

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        if self.global_prune:
            self.total_removed = self.prune_func(self)
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue
                    mask = self.masks[name]

                    # prune
                    new_mask = self.prune_func(self, mask, weight, name)
                    removed = self.name2nonzeros[name] - new_mask.sum().item()
                    self.total_removed += removed
                    self.name2removed[name] = removed
                    self.masks[name][:] = new_mask

        name2regrowth = self.calc_growth_redistribution()
        if self.global_growth:
            total_nonzero_new = self.growth_func(
                self, self.total_removed + self.adjusted_growth
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
        if self.total_nonzero > 0 and self.verbose:
            logging.info(
                "Nonzero before/after: {0}/{1}. Growth adjustment: {2:.2f}.".format(
                    self.total_nonzero, total_nonzero_new, self.adjusted_growth
                )
            )
