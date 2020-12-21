import torch.nn as nn
import torch.nn.functional as F

from models.benchmark import SparseSpeedupBench


class AlexNet(nn.Module):
    """AlexNet with batch normalization and without pooling.

    This is an adapted version of AlexNet as taken from
    SNIP: Single-shot Network Pruning based on Connection Sensitivity,
    https://arxiv.org/abs/1810.02340

    There are two different version of AlexNet:
    AlexNet-s (small): Has hidden layers with size 1024
    AlexNet-b (big):   Has hidden layers with size 2048

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(
        self,
        config="s",
        num_classes=1000,
        bench_model: bool = False,
    ):
        super(AlexNet, self).__init__()
        self.feats = []
        self.densities = []
        self.bench = None if not bench_model else SparseSpeedupBench()

        factor = 1 if config == "s" else 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024 * factor),
            nn.BatchNorm1d(1024 * factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024 * factor, 1024 * factor),
            nn.BatchNorm1d(1024 * factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024 * factor, num_classes),
        )

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
