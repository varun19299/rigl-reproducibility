from dataclasses import dataclass

import torch
import torch.optim as optim


@dataclass
class Decay(object):
    def step(self):
        raise NotImplementedError

    def get_dr(self):
        raise NotImplementedError


class CosineDecay(Decay):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """

    def __init__(
        self, prune_rate: float, T_max: int, eta_min: float = 0.0, last_epoch: int = -1
    ):
        super().__init__()
        self._step = 0
        self.T_max = T_max

        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self):
        if self._step < self.T_max:
            self.cosine_stepper.step()
            self._step += 1

    def get_dr(self):
        return self.sgd.param_groups[0]["lr"]


class LinearDecay(Decay):
    """Anneals the pruning rate linearly with each step."""

    def __init__(self, prune_rate, T_max):
        super().__init__()

        self._step = 0
        self.T_max = T_max

        self.decrement = prune_rate / float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        if self._step < self.T_max:
            self.current_prune_rate -= self.decrement
            self._step += 1

    def get_dr(self):
        return self.current_prune_rate
