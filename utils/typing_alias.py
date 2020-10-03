"""
Meant to be imported as
from utils.typing_helper import *

To ease # imports for typing.
"""

__all__ = [
    "TYPE_CHECKING",
    "Any",
    "Array",
    "DataLoader",
    "Decay",
    "Dict",
    "List",
    "lr_scheduler",
    "Masking",
    "nn.Module",
    "optim",
    "SummaryWriter",
    "Tensor",
    "tqdm",
    "Tuple",
    "Union",
]

from numpy import ndarray as Array
from sparselearning.core import Masking
from sparselearning.funcs.decay import Decay
from typing import Dict, List, Any, Tuple, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor, nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm