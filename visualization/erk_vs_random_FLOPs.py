"""
Run as:
python visualization/erk_vs_random_FLOPs.py
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sparselearning.counting.inference_train_FLOPs import RigL_train_FLOPs, wrn_22_2_FLOPs, resnet50_FLOPs

# GGplot (optional)
# plt.style.use("ggplot")

# Matplotlib font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Matplotlib line thickness
LINE_WIDTH = 4
ALPHA = 0.9

registry = {"wrn-22-2": wrn_22_2_FLOPs, "resnet50": resnet50_FLOPs}


def FLOPs_vs_sparsity(model: str = "wrn-22-2"):
    """
    FLOPs vs sparsity for Random, ERK initializations
    :param model: model name (wrn-22-2, resnet50) to use.
    :type model: str
    """
    assert model in ["wrn-22-2", "resnet50"], f"Model {model} not found"

    dense_FLOPs = registry[model](density=1.0)
    dense_train_FLOPs = 3 * dense_FLOPs  # gradient of param and activation
    print(f"WRN-22-2 Dense FLOPS: {dense_FLOPs:,} \n")

    # Masking
    interval = 100

    train_FLOPs_dict = {"Random": [], "ERK": [], "density": []}

    for density in np.linspace(0.01, 0.6, num=15):
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
            rigl_train_FLOPs = RigL_train_FLOPs(sparse_FLOPs, dense_FLOPs, interval)

            print(
                f"RigL {init_name.capitalize()} Density: {density:.3f} Train FLOPs:{rigl_train_FLOPs:,} Proportion:{rigl_train_FLOPs / dense_train_FLOPs:.4f}"
            )
            train_FLOPs_dict[init_name].append(rigl_train_FLOPs)

        print("-----------\n")

    plt.figure(figsize=(4, 6))
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

    plt.xlabel("Density (1-sparsity)")
    plt.ylabel("Train FLOPs")

    plt.subplots_adjust(left=0.15, bottom=0.125)
    plt.grid()

    plt.savefig(
        f"outputs/plots/{model}_ERK_vs_Random_train_FLOPs.pdf", dpi=150,
    )

    plt.show()


def accuracy_vs_FLOPs():
    """
    Plot test accuracy vs FLOPs for ERK, Random init
    """
    wrn_22_2_dense_FLOPs = wrn_22_2_FLOPs(density=1.0)
    wrn_22_2_dense_train_FLOPs = (
        3 * wrn_22_2_dense_FLOPs
    )  # gradient of param and activation

    resnet50_dense_FLOPs = resnet50_FLOPs(density=1.0)
    resnet50_dense_train_FLOPs = (
        3 * resnet50_dense_FLOPs
    )  # gradient of param and activation

    columns = ["Model", "Init", "FLOPs", "Test Acc Mean", "Test Acc Std"]
    df = pd.DataFrame(columns=columns)

    # TODO: can we fetch this directly from W&B?
    df.loc[0] = [
        "wrn-22-2",
        "Random",
        RigL_train_FLOPs(wrn_22_2_FLOPs("random", 0.1), wrn_22_2_dense_train_FLOPs),
        91.71666666666665,
        0.18009256878986557,
    ]
    df.loc[1] = [
        "wrn-22-2",
        "Random",
        RigL_train_FLOPs(wrn_22_2_FLOPs("random", 0.2), wrn_22_2_dense_train_FLOPs),
        92.60666666666667,
        0.3100537587795598,
    ]
    df.loc[2] = [
        "wrn-22-2",
        "Random",
        RigL_train_FLOPs(wrn_22_2_FLOPs("random", 0.5), wrn_22_2_dense_train_FLOPs),
        93.26666666666667,
        0.07234178138069844,
    ]

    df.loc[3] = [
        "wrn-22-2",
        "ERK",
        df.loc[0]["FLOPs"],
        91.43,
        0.015,
    ]
    df.loc[4] = [
        "wrn-22-2",
        "ERK",
        df.loc[1]["FLOPs"],
        92.22,
        0.1189,
    ]
    df.loc[5] = ["wrn-22-2", "ERK", df.loc[2]["FLOPs"], 93.28, 0.1545]

    df.loc[6] = [
        "resnet50",
        "Random",
        RigL_train_FLOPs(resnet50_FLOPs("random", 0.1), resnet50_dense_train_FLOPs),
        71.769513686498,
        0.3318907357554385,
    ]
    df.loc[7] = [
        "resnet50",
        "Random",
        RigL_train_FLOPs(resnet50_FLOPs("random", 0.2), resnet50_dense_train_FLOPs),
        73.53639205296834,
        0.04310600094182424,
    ]
    df.loc[8] = [
        "resnet50",
        "Random",
        RigL_train_FLOPs(resnet50_FLOPs("random", 0.5), resnet50_dense_train_FLOPs),
        74.27149415016174,
        0.3129347072329389,
    ]

    df.loc[9] = [
        "resnet50",
        "ERK",
        df.loc[6]["FLOPs"],
        71.07,
        0.3936,
    ]
    df.loc[10] = [
        "resnet50",
        "ERK",
        df.loc[7]["FLOPs"],
        71.98,
        0.3021,
    ]
    df.loc[11] = [
        "resnet50",
        "ERK",
        df.loc[8]["FLOPs"],
        74.18,
        0.4284,
    ]

    WIDTH = 0.3

    plt.figure(figsize=(8, 6))

    sub_df = df.loc[df["Model"] == "wrn-22-2"]

    erk_sub_df = sub_df.loc[sub_df["Init"] == "ERK"]
    plt.bar(
        np.arange(1, 4) - WIDTH / 2,
        erk_sub_df["Test Acc Mean"],
        width=WIDTH,
        yerr=erk_sub_df["Test Acc Std"],
    )

    random_sub_df = sub_df.loc[sub_df["Init"] == "Random"]
    plt.bar(
        np.arange(1, 4) + WIDTH / 2,
        random_sub_df["Test Acc Mean"],
        width=WIDTH,
        yerr=random_sub_df["Test Acc Std"],
    )

    plt.xticks(np.arange(1, 4), [f"{x:.2e}" for x in erk_sub_df["FLOPs"]])
    plt.ylim(88, 94)

    plt.legend(["ERK", "Random"])
    plt.xlabel("Train FLOPs")
    plt.ylabel("Accuracy (Test)")

    plt.savefig(
        "outputs/plots/wrn-22-2_ERK_vs_Random_acc_vs_train_FLOPs.pdf", dpi=150,
    )

    plt.show()

    plt.figure(figsize=(8, 6))

    sub_df = df.loc[df["Model"] == "resnet50"]

    erk_sub_df = sub_df.loc[sub_df["Init"] == "ERK"]
    plt.bar(
        np.arange(1, 4) - WIDTH / 2,
        erk_sub_df["Test Acc Mean"],
        width=WIDTH,
        yerr=erk_sub_df["Test Acc Std"],
    )

    random_sub_df = sub_df.loc[sub_df["Init"] == "Random"]
    plt.bar(
        np.arange(1, 4) + WIDTH / 2,
        random_sub_df["Test Acc Mean"],
        width=WIDTH,
        yerr=random_sub_df["Test Acc Std"],
    )

    plt.xticks(np.arange(1, 4), [f"{x:.2e}" for x in erk_sub_df["FLOPs"]])
    plt.ylim(67, 75)

    plt.legend(["ERK", "Random"])
    plt.xlabel("Train FLOPs")
    plt.ylabel("Accuracy (Test)")

    plt.savefig(
        "outputs/plots/resnet50_ERK_vs_Random_acc_vs_train_FLOPs.pdf", dpi=150,
    )

    plt.show()


if __name__ == "__main__":
    FLOPs_vs_sparsity("wrn-22-2")
    FLOPs_vs_sparsity("resnet50")
    accuracy_vs_FLOPs()
