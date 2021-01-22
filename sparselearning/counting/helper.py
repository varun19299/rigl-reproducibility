from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def get_pre_activations_dict(net: "nn.Module", input_tensor: "Tensor")-> "Dict[str, Tensor]":
    """
    Find pre-activation dict for every possible module in network

    :param net: Pytorch model
    :type net: nn.Module
    :param input_tensor: input tensor, supports only single input
    :type input_tensor: Tensor
    :return: dictionary mapping layers to pre-activations
    :rtype: Dict[str, Tensor]
    """
    """
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

    device_ll = []
    for name, weight in net.named_parameters():
        device_ll.append(weight.device)
    assert len(set(device_ll)) == 1, "No support for multi-device modules yet!"
    device = device_ll[0]

    net(input_tensor.to(device))

    return activation_dict
