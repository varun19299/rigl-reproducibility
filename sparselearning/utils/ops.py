import torch


def random_perm(a: torch.Tensor) -> torch.Tensor:
    idx = torch.randperm(a.nelement())
    return a.reshape(-1)[idx].reshape(a.shape)
