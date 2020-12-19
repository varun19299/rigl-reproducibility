from counting.ops import get_inference_FLOPs
from copy import copy, deepcopy
from dataclasses import dataclass, field

import logging
import math
import numpy as np

# Sparse learning funcs
from sparselearning.funcs.grow import registry as grow_registry
from sparselearning.funcs.prune import registry as prune_registry
from sparselearning.funcs.redistribute import registry as redistribute_registry
from sparselearning.funcs.init_scheme import registry as init_registry

import torch
import torch.nn as nn
import torch.optim as optim

from typing import TYPE_CHECKING
from utils.smoothen_value import AverageValue

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

    def load_state_dict(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def state_dict(self):
        _state_dict = {
            "variance_dict": self.variance_dict,
            "zeros_dict": self.zeros_dict,
            "nonzeros_dict": self.nonzeros_dict,
            "removed_dict": self.removed_dict,
            "total_variance": self.total_variance,
            "total_zero": self.total_zero,
            "total_nonzero": self.total_nonzero,
            "total_removed": self.total_removed,
        }
        return _state_dict

    @property
    def total_density(self):
        if not (self.total_zero + self.total_nonzero):
            return 0.0
        return self.total_nonzero / (self.total_zero + self.total_nonzero)

    def __repr__(self):
        _str_dict = {
            "total_variance": self.total_variance,
            "total_zero": self.total_zero,
            "total_nonzero": self.total_nonzero,
            "total_removed": self.total_removed,
            "total_density": self.total_density,
        }

        _str = "LayerStats("
        for e, (name, value) in enumerate(_str_dict.items()):
            if e == len(_str_dict) - 1:
                _str += f"{name}={value})"
            else:
                _str += f"{name}={value}, "

        return _str


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

    density: float = 0.1  # Sparsity = 1 - density
    sparse_init: str = "random"  # see sparselearning/funcs/init_scheme.py

    dense_gradients: bool = False

    prune_mode: str = "magnitude"
    growth_mode: str = "momentum"
    redistribution_mode: str = "momentum"

    # global growth/prune state
    prune_threshold: float = 0.001
    growth_threshold: float = 0.001
    growth_increment: float = 0.2
    increment: float = 0.2
    tolerance: float = 1e-6

    # mask & module
    masks: "Dict[str, Tensor]" = field(default_factory=dict)
    module: "nn.Module" = None  # Pytorch module

    # stats
    mask_step: int = 0

    def __post_init__(self):
        self.baseline_nonzero = 0
        self.total_params = 0

        # Growth adjustment
        self.adjusted_growth = 0
        self.adjustments = []

        self.name2prune_rate = {}
        self.stats = LayerStats()

        # FLOPs
        self._dense_FLOPs = None
        self._input_size = (1, 3, 32, 32)
        self._inference_FLOPs_collector = AverageValue()

        # Assertions
        assert (
            self.sparse_init in init_registry
        ), f"Sparse init {self.sparse_init} not found. Available {init_registry.keys()}"
        assert (
            self.growth_mode in grow_registry.keys()
        ), f"Available growth modes: {','.join(grow_registry.keys())}"

        assert (
            self.prune_mode in prune_registry.keys()
        ), f"Available prune modes: {','.join(prune_registry.keys())}"
        assert (
            self.redistribution_mode in redistribute_registry.keys()
        ), f"Available redistribute modes: {','.join(redistribute_registry.keys())}"

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

    def add_module(self, module, lottery_mask_path: "Path" = None):
        """
        Store dict of parameters to mask
        """
        self.module = module
        logging.info(f"Dense FLOPs {self.dense_FLOPs:,}")
        for name, weight in self.module.named_parameters():
            self.masks[name] = torch.zeros_like(
                weight, dtype=torch.float32, requires_grad=False
            )

        # Send to appropriate device, same as weights
        self.to_module_device_()

        # Remove bias, batchnorms
        logging.info("Removing biases...")
        self.remove_weight_partial_name("bias")
        logging.info("Removing 2D batch norms...")
        self.remove_type(nn.BatchNorm2d)
        logging.info("Removing 1D batch norms...")
        self.remove_type(nn.BatchNorm1d)

        # Call init
        self.init(lottery_mask_path)
        logging.info(f"Inference (Sparse) FLOPs (at init) {self.inference_FLOPs:,}")

    def adjust_prune_rate(self):
        """
        Modify prune rate for layers with low sparsity
        """
        for name, mask in self.masks.items():
            self.name2prune_rate[name] = self.prune_rate

            sparsity = self.stats.zeros_dict[name] / mask.numel()
            if sparsity < 0.2:
                # determine if matrix is relatively dense but still growing
                expected_variance = 1.0 / len(self.stats.variance_dict.keys())
                actual_variance = self.stats.variance_dict[name]
                expected_vs_actual = expected_variance / actual_variance

                # if weights aren't steady yet, i.e., can change significantly
                if expected_vs_actual < 1.0:
                    # growing
                    self.name2prune_rate[name] = min(
                        sparsity, self.name2prune_rate[name]
                    )

    @torch.no_grad()
    def apply_mask(self):
        """
        Applies boolean mask to modules
        """
        for name, weight in self.module.named_parameters():
            if name in self.masks:
                weight.data = weight.data * self.masks[name]

    @torch.no_grad()
    def apply_mask_gradients(self):
        """
        Applies boolean mask to modules's gradients
        """
        for name, weight in self.module.named_parameters():
            if name in self.masks:
                weight.grad = weight.grad * self.masks[name]

    @property
    def avg_inference_FLOPs(self):
        self._inference_FLOPs_collector.add_value(self.inference_FLOPs)
        return self._inference_FLOPs_collector.smooth

    def calc_growth_redistribution(self):
        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.stats.variance_dict:
                num_remove = self.stats.removed_dict[name]
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

    @property
    def dense_FLOPs(self):
        """
        Calculates dense inference FLOPs of the model
        """
        if not self._dense_FLOPs:
            self._dense_FLOPs = get_inference_FLOPs(self, torch.rand(*self._input_size))
            return self._dense_FLOPs
        else:
            return self._dense_FLOPs

    @property
    def inference_FLOPs(self):
        """
        Calculates dense inference FLOPs of the model
        """
        return get_inference_FLOPs(self, torch.rand(*self._input_size))

    @torch.no_grad()
    def init(self, lottery_mask_path: "Path"):
        # Performs weight initialization
        self.sparsify(lottery_mask_path=lottery_mask_path)
        self.to_module_device_()
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

        self.stats.total_nonzero = self.baseline_nonzero
        self.stats.total_zero = self.total_params - self.baseline_nonzero
        logging.info(f"Total parameters after removed layers: {total_size}.")
        logging.info(
            f"Total parameters under sparsity level of {self.density}: {self.baseline_nonzero}"
        )
        logging.info(
            f"Achieved sparsity at init (w/o BN, bias): {self.baseline_nonzero/self.total_params:.4f}"
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
            nonzeros_dict[name] = (mask == 1).sum().int().item()
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
        return grow_registry[self.growth_mode]

    @property
    def global_growth(self):
        return "global" in self.growth_mode

    @property
    def global_prune(self):
        return "global" in self.prune_mode

    def get_momentum_for_weight(self, weight):
        """
        Return momentum from optimizer (SGD or Adam)
        """
        momentum = []
        # Adam
        if "exp_avg" in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]["exp_avg"]
            adam_m2 = self.optimizer.state[weight]["exp_avg_sq"]
            momentum = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        # SGD
        elif "momentum_buffer" in self.optimizer.state[weight]:
            momentum = self.optimizer.state[weight]["momentum_buffer"]

        return momentum

    def load_state_dict(self, *initial_data, **kwargs):
        for state_dict in initial_data:
            for key, value in state_dict.items():
                if key == "stats":
                    self.stats.load_state_dict(value)
                else:
                    setattr(self, key, value)
        for key, value in kwargs.items():
            if key == "stats":
                self.stats.load_state_dict(kwargs[key])
            else:
                setattr(self, key, kwargs[key])

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

    def __repr__(self):
        _str_dict = {
            "baseline_nonzero": self.baseline_nonzero,
            "density": self.density,
            "dense_gradients": self.dense_gradients,
            "growth_increment": self.growth_increment,
            "growth_mode": self.growth_mode,
            "growth_threshold": self.growth_threshold,
            "increment": self.increment,
            "layer_names": self.masks.keys(),
            "mask_step": self.mask_step,
            "prune_mode": self.prune_mode,
            "prune_threshold": self.prune_threshold,
            "redistribution_mode": self.redistribution_mode,
            "sparse_init": self.sparse_init,
            "stats": self.stats,
            "tolerance": self.tolerance,
            "total_params": self.total_params,
        }

        _str = "Masking("
        for e, (name, value) in enumerate(_str_dict.items()):
            if e == len(_str_dict) - 1:
                _str += f"{name}={value})"
            else:
                _str += f"{name}={value}, "

        return _str

    @torch.no_grad()
    def reset_momentum(self):
        for name, weight in self.module.named_parameters():
            # Skip modules we aren't masking
            if name not in self.masks:
                continue

            param_state = self.optimizer.state[weight]
            mask = self.masks[name]

            # mask the momentum matrix
            # Adam
            if "exp_avg" in param_state:
                param_state["exp_avg"] *= mask
                param_state["exp_avg_sq"] *= mask

            # SGD
            elif "momentum_buffer" in param_state:
                param_state["momentum_buffer"] *= mask

    def sparsify(self, **kwargs):
        init_registry[self.sparse_init](self, **kwargs)

    def state_dict(self) -> "Dict":
        # Won't store hyperparams here
        _state_dict = {
            "baseline_nonzero": self.baseline_nonzero,
            "masks": self.masks,
            "stats": self.stats.state_dict(),
            "mask_step": self.mask_step,
            "total_params": self.total_params,
        }
        return _state_dict

    def step(self):
        """
        Performs a masking step
        """
        self.optimizer.step()
        self.apply_mask()

        if not self.dense_gradients:
            self.reset_momentum()

        # Get updated prune rate
        if self.prune_rate_decay.mode == "cumulative":
            current_sparsity = (
                1 - self.stats.total_density
            )  # Useful for pruning where we want a target sparsity
            self.prune_rate_decay.step(self.mask_step, current_sparsity)
        else:
            self.prune_rate_decay.step(self.mask_step)

        self.mask_step += 1

    def to_module_device_(self):
        """
        Send to module's device
        """
        for name, weight in self.module.named_parameters():
            if name in self.masks:
                device = weight.device
                self.masks[name] = self.masks[name].to(device)

    @torch.no_grad()
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
                # Skip modules we aren't masking
                if name not in self.masks:
                    continue

                mask = self.masks[name]

                # prune
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.stats.nonzeros_dict[name] - int(new_mask.sum().item())
                self.stats.total_removed += removed
                self.stats.removed_dict[name] = removed
                self.masks[name] = new_mask

        if self.growth_mode == "none":
            total_nonzero_new = self.stats.total_nonzero - self.stats.total_removed

        elif self.global_growth:
            total_nonzero_new = self.growth_func(
                self, self.stats.total_removed + self.adjusted_growth
            )
        else:
            if self.redistribution_mode not in ["nonzero", "none"]:
                name2regrowth = self.calc_growth_redistribution()

            for name, weight in self.module.named_parameters():
                # Skip modules we aren't masking
                if name not in self.masks:
                    continue

                new_mask = self.masks[name].data.bool()

                # growth
                if self.redistribution_mode not in ["nonzero", "none"]:
                    num_growth = name2regrowth[name]
                else:
                    feedback = self.adjustments[-1] if self.adjustments else 0
                    num_growth = self.stats.removed_dict[name]  # + feedback

                new_mask = self.growth_func(self, name, new_mask, num_growth, weight)
                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                total_nonzero_new += new_nonzero

        self.apply_mask()

        if not self.dense_gradients:
            self.reset_momentum()
            self.apply_mask_gradients()

        self.mask_step += 1

        # Some growth techniques and redistribution are probablistic
        # we might not grow enough weights or too much weights
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

        # Update stats
        self.gather_statistics()

    def update_connections(self):
        self.truncate_weights()
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            # debug logged
            self.print_nonzero_counts()
