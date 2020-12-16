from matplotlib import pyplot as plt
import numpy as np
from utils.counting_ops import resnet50_FLOPs, RigL_FLOPs, wrn_22_2_FLOPs

# Matplotlib font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Matplotlib line thickness
LINE_WIDTH = 2
ALPHA = 0.9

registry = {"wrn-22-2": wrn_22_2_FLOPs, "resnet50": resnet50_FLOPs}


def main(model: str = "wrn-22-2"):
    assert model in ["wrn-22-2", "resnet50"], f"Model {model} not found"

    dense_FLOPs = registry[model](density=1.0)
    dense_train_FLOPs = 3 * dense_FLOPs  # gradient of param and activation
    print(f"WRN-22-2 Dense FLOPS: {dense_FLOPs:,} \n")

    # Masking
    interval = 100

    train_FLOPs_dict = {"Random": [], "ERK": [], "density": []}

    for density in np.linspace(0.01, 0.6, num=20):
        Random_FLOPs = registry[model]("random", density)
        ERK_FLOPs = registry[model]("erdos-renyi-kernel", density)

        print(
            f"Random Density: {density:.3f} Inference FLOPs:{Random_FLOPs:,} Proportion:{Random_FLOPs / dense_FLOPs:.4f}"
        )
        print(
            f"ERK Density: {density:.3f} Inference FLOPs:{ERK_FLOPs:,} Proportion:{ERK_FLOPs / dense_FLOPs:.4f}"
        )

        train_FLOPs_dict["density"].append(density)

        for sparse_FLOPs, init_name in zip(
            [Random_FLOPs, ERK_FLOPs], ["Random", "ERK"]
        ):
            rigl_train_FLOPs = RigL_FLOPs(sparse_FLOPs, dense_FLOPs, interval)

            print(
                f"RigL {init_name.capitalize()} Density: {density:.3f} Train FLOPs:{rigl_train_FLOPs:,} Proportion:{rigl_train_FLOPs / dense_train_FLOPs:.4f}"
            )
            train_FLOPs_dict[init_name].append(rigl_train_FLOPs)

        print("-----------\n")

    plt.figure(figsize=(6, 5))
    plt.plot(
        train_FLOPs_dict["density"],
        train_FLOPs_dict["ERK"],
        linewidth=LINE_WIDTH,
        alpha=ALPHA,
    )
    plt.plot(
        train_FLOPs_dict["density"],
        train_FLOPs_dict["Random"],
        linewidth=LINE_WIDTH,
        alpha=ALPHA,
    )

    plt.legend(["ERK", "Random"])
    # plt.grid()

    plt.xlabel("Density (1-sparsity)")
    plt.ylabel("Train FLOPs")

    plt.savefig(
        f"outputs/plots/{model}_ERK_vs_Random_train_FLOPs.pdf", dpi=150,
    )

    plt.show()


if __name__ == "__main__":
    main("wrn-22-2")
    main("resnet50")
