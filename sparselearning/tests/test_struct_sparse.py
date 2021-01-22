import pytest
import torch
from einops import repeat
from torch import optim
from typing import TYPE_CHECKING

from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import CosineDecay

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *

def is_channel_sparse(mask: "Masking")-> bool:
    """
    Checks if the conv mask is channel-wise sparse.

    :param mask: Masking instance
    :type mask: Masking
    :return: True if channel-wise sparse
    :rtype: bool
    """
    c_in, c_out, h, w = mask.shape

    blocked = repeat(mask[:, :, 0, 0], "c_in c_out -> c_in c_out h w", h=h, w=w)
    return bool((blocked == mask).all())


@pytest.mark.parametrize(
    "init_scheme",
    [
        "struct-erdos-renyi",
        "struct-erdos-renyi-kernel",
        "struct-random",
    ],
)
def test_struct_init(init_scheme: str):
    """
    Test structured sparsity for various init schemes

    :param init_scheme: Random/ER/ERK
    :type init_scheme: str
    """
    model_class, args = model_registry["resnet50"]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    masking = Masking(
        optimizer,
        decay,
        redistribution_mode="none",
        sparse_init=init_scheme,
        density=0.5,
    )
    masking.add_module(model)
    masking.gather_statistics()
    masking.adjust_prune_rate()

    for mask in masking.mask_dict.values():
        assert is_channel_sparse(mask)


@pytest.mark.parametrize(
    "prune_mode", ["struct-magnitude-max", "struct-magnitude-mean"]
)
@pytest.mark.parametrize(
    "growth_mode", ["struct-absolute-gradient-min", "struct-absolute-gradient-mean"]
)
def test_struct_prune_growth(prune_mode, growth_mode):
    """
    Test structured sparsity across prune, growth modes.
    See sparselearning.funcs.prune,growth

    :param prune_mode: prune mode
    :type prune_mode: str
    :param growth_mode: growth mode
    :type growth_mode: str
    """
    model_class, args = model_registry["resnet50"]
    model = model_class(*args)
    decay = CosineDecay()
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    masking = Masking(
        optimizer,
        decay,
        redistribution_mode="none",
        sparse_init="struct-random",
        prune_mode=prune_mode,
        growth_mode=growth_mode,
        density=0.5,
    )

    masking.add_module(model)
    masking.gather_statistics()
    masking.adjust_prune_rate()

    inp = torch.randn(16, 3, 32, 32)

    optimizer.zero_grad()
    loss = model(inp).abs().mean()
    loss.backward()

    masking.step()
    masking.update_connections()

    for mask in masking.mask_dict.values():
        assert is_channel_sparse(mask)
