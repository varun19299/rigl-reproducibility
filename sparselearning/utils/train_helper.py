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


def get_optimizer(model: "nn.Module", **kwargs) -> "Union[optim, Tuple[lr_scheduler]]":
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
            parameters = add_weight_decay(model, weight_decay)
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


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
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
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


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
) -> "Union[nn.Module, optim, Masking, int, int, float]":
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pth_files = list(ckpt_dir.glob("epoch_*.pth"))

    # Defaults
    epoch = 0
    step = 0
    best_val_loss = 1e6

    if not resume or not pth_files:
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
