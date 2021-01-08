import torch


def random_perm(a: torch.Tensor) -> torch.Tensor:
    """
    Random shuffle a tensor
    """
    idx = torch.randperm(a.nelement())
    return a.reshape(-1)[idx].reshape(a.shape)
