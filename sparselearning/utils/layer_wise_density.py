"""
Utility for layer wise density plots.
"""
from matplotlib import pyplot as plt
import numpy as np
import wandb
from typing import TYPE_CHECKING
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *

def _get_density_ll(masking: "Masking")->"List[float]":
    non_zero_ll = np.array(list(masking.stats.nonzeros_dict.values()))
    zero_ll = np.array(list(masking.stats.zeros_dict.values()))
    density_ll = non_zero_ll / (non_zero_ll + zero_ll)

    return density_ll

def plot(masking: "Masking", mplot: plt) -> plt:
    """
    Plot layer wise density bar plot.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :param mplot: matplotlib object
    :type mplot: pyplot
    :return: matplotlib plot
    :rtype: pyplot
    """

    density_ll = _get_density_ll(masking)
    bin_ll = np.arange(len(density_ll)) + 1
    width = 0.8

    mplot.clf()
    mplot.bar(bin_ll, density_ll, width, color="b")

    # Gets too crowded when including layer names
    # layer_name_ll = list(masking.masks.keys())
    # plt.xticks(bin_ll, layer_name_ll)

    mplot.ylabel("Density")
    mplot.xlabel("Layer Number")

    return mplot


def wandb_bar(masking: "Masking")->wandb.plot.bar:
    """
    Plot layer wise density as W&B bar plot.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :return: W&B bar plot
    :rtype: wandb.plot.bar
    """
    density_ll = _get_density_ll(masking)
    label_ll = list(masking.stats.nonzeros_dict.keys())

    data = [[label, density] for (label, density) in zip(label_ll, density_ll)]
    table = wandb.Table(data=data, columns=["layer name", "density"])
    return wandb.plot.bar(table, "layer name", "density", title="Layer-wise Density")


def plot_as_image(masking: "Masking") -> "Array":
    """
    Plot layer wise density as bar plot figure.

    :param masking: Masking instance
    :type masking: sparselearning.core.Masking
    :return: Numpy array representing figure (H, W, 3)
    :rtype: np.ndarray
    """
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    density_ll = _get_density_ll(masking)
    bin_ll = np.arange(len(density_ll)) + 1
    width = 0.8

    ax.bar(bin_ll, density_ll, width, color="b")

    ax.set_ylabel("Density")
    ax.set_xlabel("Layer Number")

    canvas.draw()  # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        height, width, 3
    )

    return image


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

    image = plot_as_image(mask)
    plt.imshow(image)
    plt.show()
