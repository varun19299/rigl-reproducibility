"""
Get train, val and test dataloaders

See registry dict for available dataset options.
"""
import logging
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


@dataclass
class DatasetSplitter(Dataset):
    """
    This splitter makes sure that we always use the same training/validation split
    """
    parent_dataset: Dataset
    split: "slice" = slice(None, None)
    index_map: "Array" = np.array([0])

    def __post_init__(self):
        if len(self) <= 0:
            raise ValueError(f"Dataset split {self.split} is not positive")
        if not self.index_map.any():
            self.index_map = np.array(range(len(self.parent_dataset)), dtype=int)

    def __len__(self):
        # absolute indices
        _indices = self.split.indices(len((self.parent_dataset)))
        # compute length
        return len(range(*_indices))

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_datset"
        index = self.index_map[index + int(self.split.start or 0)]
        return self.parent_dataset[index]


def _get_CIFAR10_dataset(root: "Path") -> "Tuple[Dataset,Dataset]":
    """
    Returns CIFAR10 Dataset

    :param root: path to download to / load from
    :type root: Path
    :return: train+val, test dataset
    :rtype: Tuple[Dataset,Dataset]
    """
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

    return full_dataset, test_dataset


def _get_CIFAR100_dataset(root: "Path") -> "Tuple[Dataset,Dataset]":
    """
    Returns CIFAR100 Dataset

    :param root: path to download to / load from
    :type root: Path
    :return: train+val, test dataset
    :rtype: Tuple[Dataset,Dataset]
    """
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_dataset = datasets.CIFAR100(
        root, train=True, transform=train_transform, download=True
    )
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=test_transform, download=False
    )

    return full_dataset, test_dataset


def _get_Mini_Imagenet_dataset(root: "Path") -> "Tuple[Dataset,Dataset]":
    """
    Returns Mini-Imagenet Dataset
    (https://github.com/yaoyao-liu/mini-imagenet-tools)

    :param root: path to download to / load from
    :type root: Path
    :return: train+val, test dataset
    :rtype: Tuple[Dataset,Dataset]
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Original Inception paper reported good performance
    # with dramatic scales from 0.08 to 1.0 for cropping
    # https://discuss.pytorch.org/t/is-transforms-randomresizedcrop-used-for-data-augmentation/16716
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_dataset = datasets.ImageFolder(root / "train_val", transform=train_transform,)
    test_dataset = datasets.ImageFolder(root / "test", transform=test_transform,)

    return full_dataset, test_dataset


def _get_MNIST_dataset(root: "Path") -> "Tuple[Dataset,Dataset]":
    """
    Returns MNIST Dataset

    :param root: path to download to / load from
    :type root: Path
    :return: train+val, test dataset
    :rtype: Tuple[Dataset,Dataset]
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, transform=transform)

    return full_dataset, test_dataset


def get_dataloaders(
    name: str,
    root: "Path",
    batch_size: int = 1,
    test_batch_size: int = 1,
    validation_split: float = 0.0,
    max_threads: int = 3,
    fixed_shuffle: bool = False,
)-> "Tuple[DataLoader, DataLoader, DataLoader]":
    """
    Creates augmented train, validation, and test data loaders.

    :param name: dataset name
    :type name: str
    :param root: Path to download to / load from
    :type root: Path
    :param batch_size: mini batch for train/val split
    :type batch_size: int
    :param test_batch_size: mini batch for test split
    :type test_batch_size: int
    :param validation_split: 0-> no val
    :type validation_split: float
    :param max_threads: Max threads to use for dataloaders
    :type max_threads: int
    :param fixed_shuffle: whether to shuffle once and save shuffled indices.
    Useful when using ImageFolderDataset and want reproducible shuffling
    :type fixed_shuffle: bool
    :return: train, val, test loaders
    :rtype: Tuple[DataLoader, DataLoader, DataLoader]
    """

    assert name in registry.keys()
    full_dataset, test_dataset = registry[name](Path(root))

    # we need at least two threads in total
    max_threads = max(2, max_threads)
    val_threads = 2 if max_threads >= 6 else 1
    train_threads = max_threads - val_threads

    # Split into train and val
    train_dataset = full_dataset
    if validation_split:
        index_map = np.array(list(range(len(train_dataset))), dtype=int)

        if fixed_shuffle:
            index_map_path = Path(root) / "index_map.npy"
            if index_map_path.exists():
                logging.info(f"Loading index map from {index_map_path}")
                index_map = np.load(index_map_path, allow_pickle=True)
            else:
                np.random.shuffle(index_map)
                logging.info(f"Saving index map to {index_map_path}")
                np.save(index_map_path, index_map)

        split = int(floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, slice(None, split), index_map)
        val_dataset = DatasetSplitter(full_dataset, slice(split, None), index_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=train_threads,
        pin_memory=False,
        shuffle=True,
        multiprocessing_context="fork",
    )

    if validation_split:
        valid_loader = DataLoader(
            val_dataset,
            test_batch_size,
            shuffle=True,
            num_workers=val_threads,
            pin_memory=False,
            multiprocessing_context="fork",
        )

    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        multiprocessing_context="fork",
    )
    logging.info(f"Train dataset length {len(train_dataset)}")
    logging.info(f"Val dataset length {len(val_dataset) if validation_split else 0}")
    logging.info(f"Test dataset length {len(test_dataset)}")

    if not validation_split:
        logging.info("Running periodic eval on test data.")
        valid_loader = test_loader

    return train_loader, valid_loader, test_loader


registry = {
    "CIFAR10": _get_CIFAR10_dataset,
    "CIFAR100": _get_CIFAR100_dataset,
    "Mini-Imagenet": _get_Mini_Imagenet_dataset,
    "MNIST": _get_MNIST_dataset,
}
