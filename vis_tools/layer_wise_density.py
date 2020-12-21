from matplotlib import pyplot as plt
import numpy as np
import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def plot(masking: "Masking", plt) -> plt:
    """
    Plot layer wise density histogram
    """

    non_zero_ll = np.array(list(masking.stats.nonzeros_dict.values()))
    zero_ll = np.array(list(masking.stats.zeros_dict.values()))
    density_ll = non_zero_ll / (non_zero_ll + zero_ll)

    bin_ll = np.arange(len(density_ll)) + 1
    width = 0.8

    plt.clf()
    plt.bar(bin_ll, density_ll, width, color="b")

    # Gets too crowded when including layer names
    # layer_name_ll = list(masking.masks.keys())
    # plt.xticks(bin_ll, layer_name_ll)

    plt.ylabel("Density")
    plt.xlabel("Layer Number")

    return plt

def wandb_table(masking: "Masking"):
    non_zero_ll = np.array(list(masking.stats.nonzeros_dict.values()))
    zero_ll = np.array(list(masking.stats.zeros_dict.values()))

    density_ll = non_zero_ll / (non_zero_ll + zero_ll)
    label_ll = np.arange(len(density_ll)) + 1

    data = [[label, density] for (label, density) in zip(label_ll, density_ll)]



if __name__ == "__main__":
    from models import registry
    from sparselearning.funcs.decay import CosineDecay
    from sparselearning.core import Masking
    from torch import optim

    model_class, args = registry["resnet50"]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    mask = Masking(
        optimizer,
        decay,
        redistribution_mode="none",
        sparse_init="erdos-renyi-kernel",
        density=0.1,
    )
    mask.add_module(model)
    mask.gather_statistics()

    plt = plot(mask, plt)
    plt.show()
