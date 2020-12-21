import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from utils.typing_alias import *


class WideResNet(nn.Module):
    """Wide Residual Network with varying depth and width.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """

    def __init__(
        self,
        depth: int = 22,
        widen_factor: int = 2,
        num_classes: int = 10,
        dropRate: float = 0.3,
        bench_model: bool = False,
        small_dense_density: float = 1.0,
    ):
        """
        depth, widen_factor as described by the paper.
        droprate: float = dropout rate to apply
        bench_model: bool = benchmark model speedup (due to sparsity).
        """
        super(WideResNet, self).__init__()
        self.bench = None if not bench_model else SparseSpeedupBench()

        small_dense_multiplier = np.sqrt(small_dense_density)
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(c * small_dense_multiplier) for c in nChannels]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bench = SparseSpeedupBench() if bench_model else None
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate, bench=self.bench,
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate, bench=self.bench,
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate, bench=self.bench,
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.feats = []
        self.densities = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.bench:
            out = self.bench.forward(self.conv1, x, "conv1")
        else:
            out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class BasicBlock(nn.Module):
    """Wide Residual Network basic block

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropRate: float = 0.0,
        bench: "SparseSpeedupBench" = None,
    ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )
        self.feats = []
        self.densities = []
        self.bench = bench
        self.in_planes = in_planes

    def forward(self, x):
        conv_layers = []
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        if self.bench:
            out0 = self.bench.forward(
                self.conv1,
                (out if self.equalInOut else x),
                str(self.in_planes) + ".conv1",
            )
        else:
            out0 = self.conv1(out if self.equalInOut else x)

        out = self.relu2(self.bn2(out0))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        if self.bench:
            out = self.bench.forward(self.conv2, out, str(self.in_planes) + ".conv2")
        else:
            out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """Wide Residual Network network block which holds basic blocks.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """

    def __init__(
        self,
        nb_layers: int,
        in_planes: int,
        out_planes: int,
        block,
        stride: int,
        dropRate: float = 0.0,
        bench: "SparseSpeedupBench" = None,
    ):
        super(NetworkBlock, self).__init__()
        self.feats = []
        self.densities = []
        self.bench = bench
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                    bench=self.bench,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = WideResNet(depth=22, widen_factor=2, small_dense_density=0.5)
    summary(model, (3, 32, 32))
