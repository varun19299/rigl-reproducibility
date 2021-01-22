from functools import partial

import torch
from torch import optim

from sparselearning.counting.ops import get_inference_FLOPs
from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import MagnitudePruneDecay, CosineDecay


def Pruning_inference_FLOPs(
    dense_FLOPs: int, decay: MagnitudePruneDecay, total_steps: int = 87891
) -> float:
    """
    Inference FLOPs for Iterative Pruning, Zhu and Gupta 2018.
    Note, assumes FLOPs \propto average sparsity,
    which is approximately true in practice.

    For our report, we accurately calculate train FLOPs by evaluating FLOPs
    during each pruning iteration.

    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param decay: Pruning schedule used
    :type decay: sparselearning.funcs.decay.MagnitudePruneDecay
    :param total_steps: Total train steps
    :type total_steps: int
    :return: Pruning inference FLOPs
    :rtype: float
    """
    avg_sparsity = 0.0
    for i in range(0, total_steps):
        avg_sparsity += decay.cumulative_sparsity(i)

    avg_sparsity /= total_steps

    return dense_FLOPs * (1 - avg_sparsity)


def Pruning_train_FLOPs(
    dense_FLOPs: int, decay: MagnitudePruneDecay, total_steps: int = 87891
) -> float:
    """
    Train FLOPs for Iterative Pruning, Zhu and Gupta 2018.
    Note, assumes FLOPs \propto average sparsity,
    which is approximately true in practice.

    For our report, we accurately calculate train FLOPs by evaluating FLOPs
    during each pruning iteration.

    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param decay: Pruning schedule used
    :type decay: sparselearning.funcs.decay.MagnitudePruneDecay
    :param total_steps: Total train steps
    :type total_steps: int
    :return: Pruning train FLOPs
    :rtype: float
    """
    avg_sparsity = 0.0
    for i in range(0, total_steps):
        avg_sparsity += decay.cumulative_sparsity(i)

    avg_sparsity /= total_steps

    return 2 * dense_FLOPs * (1 - avg_sparsity) + dense_FLOPs


def RigL_train_FLOPs(
    sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100
) -> float:
    """
    Train FLOPs for Rigging the Lottery (RigL), Evci et al. 2020.

    :param sparse_FLOPs: FLOPs consumed for sparse model's forward pass
    :type sparse_FLOPs: int
    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param mask_interval: Mask update interval
    :type mask_interval: int
    :return: RigL train FLOPs
    :rtype: float
    """
    return (2 * sparse_FLOPs + dense_FLOPs + 3 * sparse_FLOPs * mask_interval) / (
        mask_interval + 1
    )


def SNFS_train_FLOPs(
    sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100
) -> int:
    """
    Train FLOPs for Sparse Networks from Scratch (SNFS), Dettmers et al. 2020.

    :param sparse_FLOPs: FLOPs consumed for sparse model's forward pass
    :type sparse_FLOPs: int
    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param mask_interval: Mask update interval
    :type mask_interval: int
    :return: SNFS train FLOPs
    :rtype: int
    """
    return 2 * sparse_FLOPs + dense_FLOPs


def SET_train_FLOPs(
    sparse_FLOPs: int, dense_FLOPs: int, mask_interval: int = 100
) -> int:
    """
    Train FLOPs for Sparse Evolutionary Training (SET), Mocanu et al. 2018.

    :param sparse_FLOPs: FLOPs consumed for sparse model's forward pass
    :type sparse_FLOPs: int
    :param dense_FLOPs: FLOPs consumed for dense model's forward pass
    :type dense_FLOPs: int
    :param mask_interval: Mask update interval
    :type mask_interval: int
    :return: SET train FLOPs
    :rtype: int
    """
    return 3 * sparse_FLOPs


def model_inference_FLOPs(
    sparse_init: str = "random",
    density: float = 0.2,
    model_name: str = "wrn-22-2",
    input_size: "Tuple" = (1, 3, 32, 32),
) -> int:
    """
    Obtain inference FLOPs for a model.

    Only for models trained with a constant FLOP sparsifying technique.
    eg: SNFS, Pruning are not supported here.

    :param sparse_init: Initialization scheme used (Random / ER / ERK)
    :type sparse_init: str
    :param density: Overall parameter density (non-zero / capacity)
    :type density: float
    :param model_name: model to use (WideResNet-22-2 or ResNet-50)
    :type model_name: str
    :param input_size: shape of input tensor
    :type input_size: Tuple
    :return:
    :rtype:
    """
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
