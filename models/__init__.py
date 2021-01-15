from models.wide_resnet import WideResNet
from models.resnet import ResNet, BasicBlock, BottleNeck

registry = {
    "resnet50": (ResNet, [BottleNeck, [3, 4, 6, 3], 100]),
    "wrn-22-2": (WideResNet, [22, 2, 10, 0.3]),
}
