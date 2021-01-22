import torch


def random_perm(a: torch.Tensor) -> torch.Tensor:
    """
    Random shuffle a tensor.

    :param a: input Tensor
    :type a: torch.Tensor
    :return: shuffled Tensor
    :rtype: torch.Tensor
    """
    idx = torch.randperm(a.nelement())
    return a.reshape(-1)[idx].reshape(a.shape)
