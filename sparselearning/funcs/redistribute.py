import torch


def momentum_redistribution(masking, name, weight, mask):
    """Calculates momentum redistribution statistics.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        mask        The binary mask. 1s indicated active weights.

    Returns:
        Layer Statistic      The unnormalized layer statistics
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.


    The calculation of redistribution statistics is the first
    step in this sparse learning library.
    """
    grad = masking.get_momentum_for_weight(weight)
    mean_magnitude = torch.abs(grad[mask.bool()]).mean().item()
    return mean_magnitude


def magnitude_redistribution(masking, name, weight, mask):
    mean_magnitude = torch.abs(weight)[mask.bool()].mean().item()
    return mean_magnitude


def nonzero_redistribution(masking, name, weight, mask):
    nonzero = (weight != 0.0).sum().item()
    return nonzero


def no_redistribution(masking, name, weight, mask):
    num_params = masking.baseline_nonzero
    n = weight.numel()
    return n / float(num_params)


def variance_redistribution(masking, name, weight, mask):
    """Return the mean variance of existing weights.

    Higher gradient variance means a layer does not have enough
    capacity to model the inputs with the current number of weights.
    Thus we want to add more weights if we have higher variance.
    If variance of the gradient stabilizes this means
    that some weights might be useless/not needed.
    """
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp.
    if "exp_avg_sq" not in masking.optimizer.state[weight]:
        print("Variance redistribution requires the adam optimizer to be run!")
        raise Exception(
            "Variance redistribution requires the adam optimizer to be run!"
        )
    iv_adam_sumsq = torch.sqrt(masking.optimizer.state[weight]["exp_avg_sq"])

    layer_importance = iv_adam_sumsq[mask.bool()].mean().item()
    return layer_importance


registry = {
    "magnitude": magnitude_redistribution,
    "momentum": momentum_redistribution,
    "nonzero": nonzero_redistribution,
    "none": no_redistribution,
    "variance_redistribution": variance_redistribution,
}
