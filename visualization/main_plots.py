import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

plt.rc("font", size=26)
plt.rc("axes", grid=True)
plt.rc("lines", linewidth=6)
plt.rc("savefig", facecolor="white")

line_alpha = 0.85
marker_edge_width = 4  # markeredgewidth
marker_size = 15
marker_alpha = 0.5
figure_size = (12, 9)

COLORS = {
    "RigL": "purple",
    "SNFS": "red",
    "SET": "blue",
    "Static": "green",
    "Pruning": "brown",
    "Dense": "black",
}

style = {"ERK": "--", "Random": "-"}

methods = list(COLORS.keys())
methods = sorted(methods)


def plot_col_vs_density(y_key, data, method, init, **plot_kwargs):
    """Plot a column for a single method + init combination."""

    dat = data.loc[data["Method"] == method]

    if method == "Dense":
        y = np.array(dat[y_key])[0]
        line = plt.axhline(
            y,
            #             **{**plot_kwargs, 'alpha':0.65},
            **plot_kwargs,
            linewidth=4,
        )
        return line

    if not list(dat["Init"])[0] is np.nan:
        dat = dat.loc[dat["Init"] == init]

    x = dat["Density"]
    y = dat[y_key]

    (line,) = plt.plot(
        x,
        y,
        style[init],
        **plot_kwargs,
    )

    return line


def plot_method(data, method, init, color):
    if method == "Pruning":
        init = "Random"

    if method in ["Dense"]:
        plot_col_vs_density(
            "Mean Acc",
            data,
            method,
            init="Random",
            color=color,
            label=method,
            alpha=line_alpha,
        )
        return

    for ykey in ["Acc seed 0", "Acc seed 1", "Acc seed 2"]:
        plot_col_vs_density(
            ykey,
            data,
            method,
            init=init,
            color=color,
            marker="x",
            markeredgewidth=marker_edge_width,
            linewidth=0,
            markersize=marker_size,
            alpha=marker_alpha,
        )
    plot_col_vs_density(
        "Mean Acc",
        data,
        method,
        init=init,
        color=color,
        label=method + (" (ERK)" if init == "ERK" else ""),
        alpha=line_alpha,
    )


def create_plot_from_spec(data, plot_spec, ylimits, name):
    plt.figure(figsize=figure_size)

    for method, init, color in plot_spec:
        plot_method(data, method, init, color)

    plt.ylim(ylimits)
    plt.xlabel("Density (1 - sparsity)")
    plt.ylabel("Accuracy (Test)")

    plt.legend()

    plt.savefig(
        f"{hydra.utils.get_original_cwd()}/outputs/plots/{name}.pdf",
        bbox_inches="tight",
    )

    plt.show()


def cifar10plots():
    dataset = "cifar10"
    csv_path = (
        f"{hydra.utils.get_original_cwd()}/outputs/csv/{dataset}_main_results.csv"
    )
    data = pd.read_csv(csv_path)

    if data["Mean Acc"][0] < 1:
        for col in ["Mean Acc", "Acc seed 0", "Acc seed 1", "Acc seed 2"]:
            data[col] *= 100

    ylimits = (90, 94)

    # fig 1a
    name = f"{dataset}_random_markers"
    plot_spec = [(method, "Random", COLORS[method]) for method in methods]
    create_plot_from_spec(data, plot_spec, ylimits, name)

    # fig 1b
    name = f"{dataset}_ERK_markers"
    plot_spec = [(method, "ERK", COLORS[method]) for method in methods]
    create_plot_from_spec(data, plot_spec, ylimits, name)

    # fig 1c
    name = f"{dataset}_2x_markers"
    plot_spec = [
        ("Dense", "Random", COLORS["Dense"]),
        ("Pruning", "Random", COLORS["Pruning"]),
        ("RigL", "ERK", COLORS["RigL"]),
        ("RigL_2x", "Random", COLORS["RigL"]),
        ("SET_2x", "Random", COLORS["SET"]),
        ("Static_2x", "Random", COLORS["Static"]),
    ]
    create_plot_from_spec(data, plot_spec, ylimits, name)


def cifar100plots():
    dataset = "cifar100"
    csv_path = (
        f"{hydra.utils.get_original_cwd()}/outputs/csv/{dataset}_main_results.csv"
    )
    data = pd.read_csv(csv_path)

    if data["Mean Acc"][0] < 1:
        for col in ["Mean Acc", "Acc seed 0", "Acc seed 1", "Acc seed 2"]:
            data[col] *= 100

    # fig 2a
    name = f"{dataset}_random_markers"
    plot_spec = [(method, "Random", COLORS[method]) for method in methods]
    create_plot_from_spec(data, plot_spec, (68, 76), name)

    # fig 2b
    name = f"{dataset}_ERKand2x_markers"
    plot_spec = [
        ("Dense", "Random", COLORS["Dense"]),
        ("Pruning", "Random", COLORS["Pruning"]),
        ("RigL", "ERK", COLORS["RigL"]),
        ("RigL_2x", "Random", "xkcd:magenta"),
        ("RigL_2x", "ERK", "xkcd:magenta"),
        ("RigL_3x", "Random", "violet"),
        ("SNFS", "ERK", COLORS["SNFS"]),
    ]
    create_plot_from_spec(data, plot_spec, (71, 75.6), name)


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    cifar10plots()
    cifar100plots()


if __name__ == "__main__":
    main()
