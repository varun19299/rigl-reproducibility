from models.alexnet import AlexNet
from models.vgg_16 import VGG16
from models.wide_resnet import WideResNet

registry = {
    "alexnet-b": (AlexNet, ["b", 10]),
    "alexnet-s": (AlexNet, ["s", 10]),
    "lenet5": (LeNet_5_Caffe, []),
    "lenet300-100": (LeNet_300_100, []),
    "vgg-c": (VGG16, ["C", 10]),
    "vgg-d": (VGG16, ["D", 10]),
    "vgg-like": (VGG16, ["like", 10]),
    "wrn-28-2": (WideResNet, [28, 2, 10, 0.3]),
    "wrn-22-2": (WideResNet, [22, 2, 10, 0.3]),
    "wrn-22-8": (WideResNet, [22, 8, 10, 0.3]),
    "wrn-16-8": (WideResNet, [16, 8, 10, 0.3]),
    "wrn-16-10": (WideResNet, [16, 10, 10, 0.3]),
}

class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

    Basic usage:
    1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
    2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
        self.bench = SparseSpeedupBench()
        self.conv_layer1 = nn.Conv2(3, 96, 3)

        if self.bench:
            outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
        else:
            outputs = self.conv_layer1(inputs)
    3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """

    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data != 0.0).sum().item() / x.numel()

    def forward(self, layer, x, layer_id):
        if not self.layer_0_idx:
            self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx:
            self.iter_idx += 1
        self.print_weights(layer.weight.data, layer)

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in / float(
            num_channels_in * batch_size
        )
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end) / 1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = (
            ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        )
        channel_sparsity_weight = sparse_channels / float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.layer_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(
            time_taken_s
            * (1.0 - channel_sparsity_weight)
            * (1.0 - channel_sparsity_input)
        )
        self.layer_timings_sparse[layer_id].append(
            time_taken_s * input_sparsity * weight_sparsity
        )

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print("\n")
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_sparse += t_sparse
            total_time_channel_sparse += t_channel_sparse

            print(
                "Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}".format(
                    layer_id, t_dense, t_channel_sparse, t_sparse
                )
            )
        self.total_timings.append(total_time_dense)
        self.total_timings_sparse.append(total_time_sparse)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)

        print("Speedups for this segment:")
        print(
            "Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x".format(
                total_time_dense,
                total_time_channel_sparse,
                total_time_dense / total_time_channel_sparse,
            )
        )
        print(
            "Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x".format(
                total_time_dense,
                total_time_sparse,
                total_time_dense / total_time_sparse,
            )
        )
        print("\n")

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print("Speedups for entire training:")
        print(
            "Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x".format(
                total_dense, total_channel_sparse, total_dense / total_channel_sparse
            )
        )
        print(
            "Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x".format(
                total_dense, total_sparse, total_dense / total_sparse
            )
        )
        print("\n")

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)

