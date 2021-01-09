"""
Cross Entropy with Label Smoothing
"""
import torch.nn.functional as F
from torch import nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def _linear_combination(
    x: "Union[float, Tensor]", y: "Union[float, Tensor]", epsilon
) -> "Union[float, Tensor]":
    """
    Affine combination of x, y
    """
    return epsilon * x + (1 - epsilon) * y


def _reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements Cross Entropy with Label Smoothing

    Source: https://github.com/pytorch/pytorch/issues/7455#issuecomment-720100742
    """

    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, log_preds: "Tensor", target: "Tensor") -> "Tensor":
        num_classes = log_preds.size()[-1]
        loss = _reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return _linear_combination(loss / num_classes, nll, self.epsilon)
