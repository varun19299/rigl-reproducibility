from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import _LRScheduler

if TYPE_CHECKING:
    from utils.typing_alias import *


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters: int, last_epoch: int = -1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    from torch import optim
    from models.alexnet import AlexNet
    from matplotlib import pyplot as plt

    model = AlexNet()

    optimizer = optim.SGD(model.parameters(), lr=1)

    warmup_scheduler = WarmUpLR(optimizer, total_iters=100)
    step_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10)

    lr = []
    for step in range(300):
        if step < warmup_scheduler.total_iters:
            warmup_scheduler.step()
            lr.append(warmup_scheduler.get_lr())
        else:
            step_scheduler.step()
            lr.append(step_scheduler.get_lr())

    plt.plot(lr)
    plt.show()
