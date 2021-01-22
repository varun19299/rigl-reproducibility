"""
python sparselearning/visualization/lr_tuning.py wandb.project="cifar10 grid lr" dataset=CIFAR10
"""
from typing import TYPE_CHECKING, List, Union

import torch
from multipledispatch import dispatch
from matplotlib import pyplot as plt
import numpy as np
from torch import nn

from models import registry as model_registry
from sparselearning.utils.model_serialization import load_state_dict

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *

# Matplotlib font sizes
TINY_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Matplotlib line thickness
LINE_WIDTH = 4
ALPHA = 0.9


@dispatch(nn.Module)
def get_layer_wise_density(model: "nn.Module") -> "Tuple[List[str], List[float]]":
    """
    Layer-wise density list

    :param model: Pytorch model
    :type model: nn.Module
    :return: layer names, density
    :rtype: Tuple[List[str], List[float]]
    """
    density_ll = []
    name_ll = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
            continue

        name_ll.append(name)

        weight = module.weight
        num_nonzero = (torch.abs(weight) > 0).sum()
        density_ll.append(num_nonzero / weight.numel())

    return name_ll, density_ll


@dispatch(str)
def get_layer_wise_density(model_path: str) -> "Tuple[List[str], List[float]]":
    """
    Layer-wise density list

    :param model_path: Path to model's ckpt
    :type model_path: str
    :return: layer names, density
    :rtype: Tuple[List[str], List[float]]
    """
    model_class, args = model_registry["wrn-22-2"]
    model = model_class(*args)

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))

    if "model" in ckpt:
        load_state_dict(model, ckpt["model"])
    elif "state_dict" in ckpt:
        load_state_dict(model, ckpt["state_dict"])
    else:
        raise KeyError

    return get_layer_wise_density(model)


if __name__ == "__main__":
    legend_ll = ["Random", "ERK", "Sparse Grad", "Sparse Mmt"]
    path_ll = [
        "outputs/CIFAR10/RigL_Random/+specific=cifar_wrn_22_2_masking,masking.density=0.1,masking.sparse_init=random,seed=0/ckpts/epoch_250.pth",
        "outputs/CIFAR10/RigL_ERK/+specific=cifar_wrn_22_2_masking,masking.density=0.1,seed=0/ckpts/epoch_250.pth",
        "outputs/CIFAR10/RigL-SG_Random/0.1/+specific=cifar10_wrn_22_2_masking,masking.print_FLOPs=True,masking.redistribution_mode=grad,masking.sparse_init=random,seed=3/ckpts/epoch_250.pth",
        "outputs/CIFAR10/RigL-SM_Random/0.1/+specific=cifar10_wrn_22_2_masking,masking.redistribution_mode=momentum,seed=3/ckpts/epoch_250.pth",
    ]
    WIDTH = 0.2
    SPACING = 0.2
    plt.figure(figsize=(14, 5))

    for e, path in enumerate(path_ll):
        name_ll, density_ll = get_layer_wise_density(path)

        plt.bar(
            np.arange(1, len(density_ll) + 1) + SPACING * (e - 1.5),
            density_ll,
            width=WIDTH,
        )
    name_ll = [name.replace("block", "b").replace("layer.", "l") for name in name_ll]
    plt.xticks(np.arange(1, len(density_ll) + 1), name_ll, rotation=25, ha="right")

    plt.legend(legend_ll)
    plt.ylabel("Density", labelpad=10)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("outputs/plots/density_dist_rand_erk_redist.pdf", dpi=150)

    plt.show()
