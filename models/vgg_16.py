import torch.nn as nn
import torch.nn.functional as F

from models.benchmark import SparseSpeedupBench

VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    "like": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "C": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        (1, 256),
        "M",
        512,
        512,
        (1, 512),
        "M",
        512,
        512,
        (1, 512),
        "M",  # tuples indicate (kernel size, output channels)
    ],
}


class VGG16(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config, num_classes=10, save_features=False, bench_model=False):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = None if not bench_model else SparseSpeedupBench()

        if config == "C" or config == "D":
            self.classifier = nn.Sequential(
                nn.Linear(
                    (512 if config == "D" else 2048), 512
                ),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )

    @staticmethod
    def make_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                kernel_size = 3
                if isinstance(v, tuple):
                    kernel_size, v = v
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                    self.densities.append((x.data != 0.0).sum().item() / x.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
