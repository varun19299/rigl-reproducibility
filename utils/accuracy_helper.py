from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from utils.typing_alias import *


def get_topk_accuracy(output: "Tensor", target: "Tensor", topk: "Tuple" = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True,)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res
