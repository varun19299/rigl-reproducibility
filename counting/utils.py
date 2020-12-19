from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def get_pre_activations_dict(net: "nn.Module", input_tensor: "Tensor"):
    """
    Find (pre)activation dict for every possible module in net
    """
    # TODO: this function invokes the warning
    # torch/nn/modules/container.py:434: UserWarning: Setting attributes on ParameterList is not supported.
    # warnings.warn("Setting attributes on ParameterList is not supported.")
    # Why?

    activation_dict = {}

    def _get_activation(name):
        def hook(model, input, output):
            # activation_dict[name] = output.detach()
            activation_dict[name] = input[0].detach()

        return hook

    for name, module in net.named_modules():
        module.register_forward_hook(_get_activation(name))

    net(input_tensor)

    return activation_dict
