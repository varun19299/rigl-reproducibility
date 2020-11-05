import logging
import os
from math import floor
import numpy as np
from pathlib import Path
import re
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

from utils.model_serialization import load_state_dict


class DatasetSplitter(Dataset):
    """This splitter makes sure that we always use the same training/validation split"""

    def __init__(
        self, parent_dataset: Dataset, split_start: int = -1, split_end: int = -1
    ):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert (
            split_start <= len(parent_dataset) - 1
            and split_end <= len(parent_dataset)
            and split_start < split_end
        ), "invalid dataset split, check bounds of split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_dataloaders(
    name: str,
    root: str,
    batch_size: int,
    test_batch_size: int,
    validation_split: float = 0.0,
    max_threads: int = 3,
):
    """Creates augmented train, validation, and test data loaders."""

    assert name in ["CIFAR10", "MNIST"]

    if name == "CIFAR10":
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )

        train_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_dataset = datasets.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root, train=False, transform=test_transform, download=False
        )
    elif name == "MNIST":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_dataset = datasets.MNIST(
            root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(root, train=False, transform=transform)

    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1

    # Split into train and val
    valid_loader = None
    val_dataset = []
    if validation_split:
        split = int(floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            num_workers=train_threads,
            pin_memory=False,
            shuffle=True,
            multiprocessing_context="fork",
        )
        valid_loader = DataLoader(
            val_dataset,
            test_batch_size,
            num_workers=val_threads,
            pin_memory=False,
            multiprocessing_context="fork",
        )
    else:
        train_dataset = full_dataset
        train_loader = DataLoader(
            full_dataset,
            batch_size,
            num_workers=max_threads,
            pin_memory=False,
            shuffle=True,
            multiprocessing_context="fork",
        )

    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        multiprocessing_context="fork",
    )
    logging.info(f"Train dataset length {len(train_dataset)}")
    logging.info(f"Val dataset length {len(val_dataset)}")
    logging.info(f"Test dataset length {len(test_dataset)}")

    if not valid_loader:
        logging.info("Running periodic eval on test data.")
        valid_loader = test_loader

    return train_loader, valid_loader, test_loader


def get_optimizer(model: "nn.Module", **kwargs) -> "Union[optim, lr_scheduler]":
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

    return optimizer, lr_scheduler


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
    logging.info(f"Epoch {epoch + 1} saving weights")

    state_dict = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "mask": mask.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
    }

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
) -> "Union[nn.Module, optim, int, int, float, int]":
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pth_files = list(ckpt_dir.glob("epoch_*.pth"))

    # Defaults
    epoch = 0
    step = 0
    best_val_loss = 1e6
    mask_steps = 0

    if not resume or not pth_files:
        logging.info(f"No checkpoint found at {ckpt_dir.resolve()}.")
        return model, optimizer, step, epoch, best_val_loss, mask_steps

    # Extract latest epoch
    latest_epoch = max([int(re.findall("\d+", file.name)[-1]) for file in pth_files])

    # Extract latest model
    model_path = list(ckpt_dir.glob(f"*_{latest_epoch}.pth"))[0]

    logging.info(f"Loading checkpoint from {model_path}.")

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    load_state_dict(model, ckpt["model"])

    epoch = ckpt.get("epoch", 1) - 1
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", "not stored")
    mask_steps = ckpt.get("mask_steps", 0)
    logging.info(f"Model has val loss of {val_loss}.")

    # Extract best loss
    best_model_path = ckpt_dir / "best_model.pth"
    if best_model_path:
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        best_val_loss = ckpt.get("val_loss", "not stored")
        logging.info(
            f"Best model has val loss of {best_val_loss} at epoch {ckpt.get('epoch',1)-1}."
        )

    return model, optimizer, step, epoch, best_val_loss, mask_steps


class SmoothenValue(object):
    "Create a smooth moving average for a value (loss, etc) using `beta`."

    def __init__(self, beta: float = 0.9):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)
