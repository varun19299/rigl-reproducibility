from models.alexnet import AlexNet
from models.vgg_16 import VGG16
from models.wide_resnet import WideResNet
from models.resnet import ResNet, BasicBlock, Bottleneck

registry = {
    "alexnet-b": (AlexNet, ["b", 10]),
    "alexnet-s": (AlexNet, ["s", 10]),
    "resnet50": (ResNet, [Bottleneck, [3, 4, 6, 3], 10])
    "vgg-c": (VGG16, ["C", 10]),
    "vgg-d": (VGG16, ["D", 10]),
    "vgg-like": (VGG16, ["like", 10]),
    "wrn-28-2": (WideResNet, [28, 2, 10, 0.3]),
    "wrn-22-2": (WideResNet, [22, 2, 10, 0.3]),
    "wrn-22-8": (WideResNet, [22, 8, 10, 0.3]),
    "wrn-16-8": (WideResNet, [16, 8, 10, 0.3]),
    "wrn-16-10": (WideResNet, [16, 10, 10, 0.3]),
}
