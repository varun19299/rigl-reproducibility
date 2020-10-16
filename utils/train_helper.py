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
            pin_memory=True,
            shuffle=True,
        )
        valid_loader = DataLoader(
            val_dataset, test_batch_size, num_workers=val_threads, pin_memory=True
        )
    else:
        train_dataset = full_dataset
        train_loader = DataLoader(
            full_dataset,
            batch_size,
            num_workers=max_threads,
            pin_memory=True,
            shuffle=True,
        )

    test_loader = DataLoader(
        test_dataset, test_batch_size, shuffle=False, num_workers=1, pin_memory=True,
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
    val_loss: float,
    step: int,
    epoch: int,
    ckpt_dir: str,
    is_min: bool = True,
):
    logging.info(f"Epoch {epoch + 1} saving weights")

    # Gen
    state_dict = {
        "step": step,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    model_path = f"model_best_epoch_{epoch}.pth" if is_min else f"epoch_{epoch}.pth"
    model_path = Path(ckpt_dir) / model_path

    torch.save(state_dict, model_path)


def load_weights(
    model: "nn.Module", optimizer: "optim", ckpt_dir: str, resume: bool = True
) -> "Union[nn.Module, optim, int, int, float, bool]":
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pth_files = list(ckpt_dir.glob("*.pth"))

    # Defaults
    epoch = 0
    step = 0
    best_val_loss = 1e6

    if not resume or not pth_files:
        logging.info(f"No checkpoint found  at {ckpt_dir}.")
        return model, optimizer, step, epoch, best_val_loss, False

    # Extract latest epoch
    latest_epoch = max([int(re.findall("\d+", file.name)[-1]) for file in pth_files])

    # Extract latest model
    model_path = list(ckpt_dir.glob(f"*_{latest_epoch}.pth"))[0]

    logging.info(f"Loading checkpoint from {model_path}.")

    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    load_state_dict(model, ckpt["state_dict"])

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", "not stored")
    logging.info(f"Model has val loss of {val_loss}.")

    # Extract best loss
    best_model_path = list(ckpt_dir.glob(f"*best_epoch*.pth"))[0]
    if best_model_path:
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        best_val_loss = ckpt.get("val_loss", "not stored")
        logging.info(f"Best model has val loss of {best_val_loss}.")

    return model, optimizer, step, epoch, best_val_loss, True


class SmoothenValue(object):
    "Create a smooth moving average for a value (loss, etc) using `beta`."

    def __init__(self, beta: float = 0.9):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = "alexnet"
    # model_name = 'vgg'
    # model_name = 'wrn'

    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0:
            print(batch_idx, "/", len(test_loader))
        with torch.no_grad():
            # if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                # print('=='*50)
                # print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        # print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        # print(feat_id, map_id, cls)
                        # print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save("./results/{0}_sparse_density_data".format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        # print(feat_id, data)
        full_contribution = data.sum()
        # print(full_contribution, data)
        contribution_per_channel = (1.0 / full_contribution) * data.sum(1)
        # print('pre', data.shape[0])
        channels = data.shape[0]
        # data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, "pre")
        data = data[idx[threshold_idx:]]
        print(data.shape, "post")

        # perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        # print(contribution_per_channel, perc, feat_id)
        # data = data[contribution_per_channel > perc]
        # print(contribution_per_channel[contribution_per_channel < perc].sum())
        # print('post', data.shape[0])
        normed_data = np.max(data / np.sum(data, 1).reshape(-1, 1), 1)
        # normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        # counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save(
            "./results/{2}_{1}_feat_data_layer_{0}".format(
                feat_id, "sparse" if sparse else "dense", model_name
            ),
            normed_data,
        )
        # plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        # plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        # plt.xlim(0.1, 0.5)
        # if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        # else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        # plt.clf()
