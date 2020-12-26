from functools import partial

import torch
from torch import optim

from sparselearning.counting.ops import get_inference_FLOPs
from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import MagnitudePruneDecay, CosineDecay


def Pruning_inference_FLOPs(
    dense_FLOPs: int, decay: MagnitudePruneDecay, total_steps: int = 87891
):
    avg_sparsity = 0.0
    for i in range(0, total_steps):
        avg_sparsity += decay.cumulative_sparsity(i)

    avg_sparsity /= total_steps

    return dense_FLOPs * (1 - avg_sparsity)


def Pruning_train_FLOPs(
    dense_FLOPs: int, decay: MagnitudePruneDecay, total_steps: int = 87891
):
    avg_sparsity = 0.0
    for i in range(0, total_steps):
        avg_sparsity += decay.cumulative_sparsity(i)

    avg_sparsity /= total_steps

    return dense_FLOPs * (1 - avg_sparsity) + 2 * dense_FLOPs


def RigL_train_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return (2 * sparse_FLOPs + dense_FLOPs + 3 * sparse_FLOPs * mask_interval) / (
        mask_interval + 1
    )


def SNFS_train_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return 2 * sparse_FLOPs + dense_FLOPs


def SET_train_FLOPs(sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100):
    return 3 * sparse_FLOPs


def model_inference_FLOPs(
    sparse_init: str = "random",
    density: float = 0.2,
    model_name: str = "wrn-22-2",
    input_size: "Tuple" = (1, 3, 32, 32),
) -> int:
    model_class, args = model_registry[model_name]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    mask = Masking(optimizer, decay, sparse_init=sparse_init, density=density)
    mask.add_module(model)

    return get_inference_FLOPs(mask, torch.rand(*input_size))


registry = {
    "RigL": RigL_train_FLOPs,
    "SET": SET_train_FLOPs,
    "SNFS": SNFS_train_FLOPs,
    "Pruning": Pruning_train_FLOPs,
}
wrn_22_2_FLOPs = partial(
    model_inference_FLOPs, model_name="wrn-22-2", input_size=(1, 3, 32, 32)
)
resnet50_FLOPs = partial(
    model_inference_FLOPs, model_name="resnet50", input_size=(1, 3, 32, 32)
)
