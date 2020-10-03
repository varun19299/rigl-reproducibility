from dataclasses import dataclass

import torch
import torch.optim as optim


@dataclass
class Decay(object):
    def step(self):
        raise NotImplementedError

    def get_dr(self, prune_rate):
        raise NotImplementedError


class CosineDecay(Decay):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """

    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        super().__init__()

        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]["lr"]


class LinearDecay(Decay):
    """Anneals the pruning rate linearly with each step."""

    def __init__(self, prune_rate, T_max):
        super().__init__()

        self.steps = 0
        self.decrement = prune_rate / float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate
