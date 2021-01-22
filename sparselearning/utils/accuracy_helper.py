"""
Implements Top-k accuracy
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def get_topk_accuracy(
    output: "Tensor", target: "Tensor", topk: "Tuple" = (1,)
) -> "List[float]":
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    :param output: predicted labels
    :type output: torch.Tensor
    :param target: groundtruth labels
    :type target: torch.Tensor
    :param topk: k for which top-k should be evaluted
    :type topk: Tuple[int]
    :return: Top-k accuracies for each value of k supplied
    :rtype: List[int]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(
            k=maxk,
            dim=1,
            largest=True,
            sorted=True,
        )
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res
