import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import torch
from torch import nn, optim

from sparselearning.utils.model_serialization import load_state_dict
from sparselearning.utils.warmup_scheduler import WarmUpLR

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def get_optimizer(model: "nn.Module", **kwargs) -> "Tuple[optim, Tuple[lr_scheduler]]":
    """
    Get model optimizer

    :param model: Pytorch model
    :type model: nn.Module
    :return: Optimizer, LR Scheduler(s)
    :rtype: Tuple[optim, Tuple[lr_scheduler]]
    """
    name = kwargs["name"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    decay_frequency = kwargs["decay_frequency"]
    decay_factor = kwargs["decay_factor"]

    if name == "SGD":
        # Pytorch weight decay erroneously includes
        # biases and batchnorms
        if weight_decay:
            logging.info("Excluding bias and batchnorm layers from weight decay.")
            parameters = _add_weight_decay(model, weight_decay)
            weight_decay = 0
        else:
            parameters = model.parameters()
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=kwargs["momentum"],
            weight_decay=weight_decay,
            nesterov=kwargs["use_nesterov"],
        )
    elif name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer.")

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, decay_frequency, gamma=decay_factor
    )

    warmup_steps = kwargs.get("warmup_steps", 0)
    warmup_scheduler = WarmUpLR(optimizer, warmup_steps) if warmup_steps else None

    return optimizer, (lr_scheduler, warmup_scheduler)


def _add_weight_decay(model, weight_decay=1e-5, skip_list=())-> "Tuple[Dict[str, float],Dict[str, float]]":
    """
    Excludes batchnorm and bias from weight decay

    :param model: Pytorch model
    :type model: nn.Module
    :param weight_decay: L2 Weight decay to use
    :type weight_decay: float
    :param skip_list: names of layers to skip
    :type skip_list: Tuple[str]
    :return: Two dictionaries, with layers to apply weight decay to.
    :rtype: Tuple[Dict[str, float],Dict[str, float]]
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Bias, BN have shape 1
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return (
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    )


def save_weights(
    model: "nn.Module",
    optimizer: "optim",
    mask: "Masking",
    val_loss: float,
    step: int,
    epoch: int,
    ckpt_dir: str,
    is_min: bool = True,
):
    """
    Save progress.

    :param model: Pytorch model
    :type model: nn.Module
    :param optimizer: model optimizer
    :type optimizer: torch.optim.Optimizer
    :param mask: Masking instance
    :type mask: sparselearning.core.Masking
    :param val_loss: Current validation loss
    :type val_loss: float
    :param step: Current step
    :type step: int
    :param epoch: Current epoch
    :type epoch: int
    :param ckpt_dir: Checkpoint directory
    :type ckpt_dir: Path
    :param is_min: Whether current model achieves least val loss
    :type is_min: bool
    """
    logging.info(f"Epoch {epoch} saving weights")

    state_dict = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    if mask:
        state_dict["mask"] = mask.state_dict()

    model_path = Path(ckpt_dir) / f"epoch_{epoch}.pth"

    torch.save(state_dict, model_path)

    if is_min:
        model_path = Path(ckpt_dir) / "best_model.pth"
        torch.save(state_dict, model_path)


def load_weights(
    model: "nn.Module",
    optimizer: "optim",
    mask: "Masking",
    ckpt_dir: str,
    resume: bool = True,
) -> "Tuple[nn.Module, optim, Masking, int, int, float]":
    """
    Load model, optimizers, mask from a checkpoint file (.pth).

    :param model: Pytorch model
    :type model: nn.Module
    :param optimizer: model optimizer
    :type optimizer: torch.optim.Optimizer
    :param mask: Masking instance
    :type mask: sparselearning.core.Masking
    :param ckpt_dir: Checkpoint directory
    :type ckpt_dir: Path
    :param resume: resume or not, if not do nothing
    :type resume: bool
    :return: model, optimizer, mask, step, epoch, best_val_loss
    :rtype: Tuple[nn.Module, optim, Masking, int, int, float]
    """
    # Defaults
    step = 0
    epoch = 0
    best_val_loss = 1e6

    if not resume:
        logging.info(f"Not resuming, training from scratch.")
        return model, optimizer, mask, step, epoch, best_val_loss

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pth_files = list(ckpt_dir.glob("epoch_*.pth"))

    if not pth_files:
        logging.info(f"No checkpoint found at {ckpt_dir.resolve()}.")
        return model, optimizer, mask, step, epoch, best_val_loss

    # Extract latest epoch
    latest_epoch = max([int(re.findall("\d+", file.name)[-1]) for file in pth_files])

    # Extract latest model
    model_path = list(ckpt_dir.glob(f"*_{latest_epoch}.pth"))[0]

    logging.info(f"Loading checkpoint from {model_path}.")

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    load_state_dict(model, ckpt["model"])

    if mask and "mask" in ckpt:
        mask.load_state_dict(ckpt["mask"])
        mask.to_module_device_()

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", "not stored")

    logging.info(f"Model has val loss of {val_loss}.")

    # Extract best loss
    best_model_path = ckpt_dir / "best_model.pth"
    if best_model_path:
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        best_val_loss = ckpt.get("val_loss", "not stored")
        logging.info(
            f"Best model has val loss of {best_val_loss} at epoch {ckpt.get('epoch',1)-1}."
        )

    return model, optimizer, mask, step, epoch, best_val_loss
