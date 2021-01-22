"""
Try testing salient dataset features:
    1. Is it downloaded?
    2. Does the loader work
    3. Does the loader have data in your desired format?
"""
import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from data import get_dataloaders, DatasetSplitter, registry


def test_splitter():
    """
    Test data splitting using DatasetSplitter
    """
    train_x = torch.rand(10, 3, 32, 32)
    train_y = torch.rand(10, 10)
    dataset = TensorDataset(train_x, train_y)
    with pytest.raises(Exception) as e_info:
        # Should raise Value error on slice
        DatasetSplitter(dataset, slice(3, 1))
        print(e_info)


@pytest.mark.parametrize("dataset", ["CIFAR10", "CIFAR100", "MNIST"])
def test_registry(dataset):
    """
    Test get_dataset functions

    :param dataset: Dataset to use
    :type dataset: str
    """
    full_dataset, test_dataset = registry[dataset](root=Path(f"datasets/{dataset}"))


def _loader_loop(loader):
    """
    Test dataloader
    """
    for x, y in tqdm(loader):
        assert len(x.shape) == 4  # NCHWW
        assert len(y.shape) == 1  # class ID


@pytest.mark.parametrize("dataset", ["CIFAR10", "CIFAR100", "MNIST"])
def test_get_loaders(dataset):
    """
    Test dataloader

    :param dataset: Dataset to use
    :type dataset: str
    """
    logging.info(f"Loading dataloaders for {dataset}")
    loaders = get_dataloaders(
        dataset,
        root=f"datasets/{dataset}",
        batch_size=128,
        test_batch_size=128,
        validation_split=0.1,
        fixed_shuffle=True,
    )
    logging.info(f"Looping through dataset {dataset}")
    for loader in loaders:
        _loader_loop(loader)
