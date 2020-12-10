import hydra
import itertools
import logging
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
from scipy.interpolate import griddata
import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def get_stats(runs, reorder: bool = True,) -> pd.DataFrame:
    """
    List all possible choices for
    masking, init, density, dataset

    We'll try matching the exhaustive caretesian product
    """
    columns = [
        "Method",
        "Init",
        "Density",
        "alpha",
        "Delta T",
        "Val Acc",
        "Test Acc",
    ]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    for e, run in enumerate(runs):
        masking = run.config["masking"]["name"]
        init = (
            "ERK"
            if run.config["masking"]["sparse_init"] == "erdos-renyi-kernel"
            else "Random"
        )
        density = run.config["masking"]["density"]
        alpha = run.config["masking"]["prune_rate"]
        deltaT = run.config["masking"]["interval"]
        val_accuracy = run.summary.val_accuracy * 100
        test_accuracy = run.summary.test_accuracy * 100

        df.loc[e] = [masking, init, density, alpha, deltaT, val_accuracy, test_accuracy]

    df = df.sort_values(by=["Method", "Init", "Density", "alpha", "Delta T"])

    if reorder:
        df = df.reset_index(drop=True)
    return df


def alpha_deltaT_plot(
    df: pd.DataFrame,
    dataset: str = "CIFAR10",
    init_ll: "List[str]" = ["ERK", "Random"],
    density_ll=[0.1, 0.2, 0.5],
):
    legend = []
    markers = ["o", "^", "s"]
    for e, (init, density) in enumerate(itertools.product(init_ll, density_ll)):
        sub_df = df.loc[(df["Init"] == init) & (df["Density"] == density)]
        print(sub_df.loc[sub_df["Test Acc"].idxmax()])

        alpha_ll = sub_df["alpha"]
        deltaT_ll = sub_df["Delta T"]
        test_accuracy_ll = sub_df["Test Acc"].to_numpy()

        # xi = np.linspace(0.1, 0.6, 50)
        # yi = np.linspace(50, 1000, 50)
        # grid_x, grid_y = np.meshgrid(xi, yi)
        #
        # triang = tri.Triangulation(alpha_ll, deltaT_ll)
        # interpolator = tri.LinearTriInterpolator(triang, test_accuracy_ll)
        # zi = interpolator(grid_x, grid_y)
        #
        # # Perform interpolation on meshgrid
        # zi = griddata(
        #     (alpha_ll, deltaT_ll), test_accuracy_ll, (grid_x, grid_y), method="cubic"
        # )
        #
        # plt.contourf(xi, yi, zi, levels=14)

        z = (test_accuracy_ll - test_accuracy_ll.min() + 1e-12) / (
            test_accuracy_ll.max() - test_accuracy_ll.min()
        )
        z *= 30
        # plt.scatter(
        #     alpha_ll, deltaT_ll, s=3 * z ** 2, c=test_accuracy_ll, cmap="plasma", alpha=0.7
        # )

        plt.scatter(
            alpha_ll,
            deltaT_ll,
            s=100,
            c=test_accuracy_ll,
            cmap="RdBu",
            alpha=0.8,
            marker=markers[e],
        )

        legend.append(rf"init={init}, $1-s$={density}")

    # plt.tricontourf(alpha_ll, deltaT_ll, test_accuracy_ll, levels=5)
    # plt.tricontour(alpha_ll, deltaT_ll, test_accuracy_ll, levels=5)
    cbar = plt.colorbar()
    cbar.set_label("Rel. Accuracy (Test)")

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\Delta T$")

    # plt.legend(legend)

    # plt.savefig(
    #     f"{hydra.utils.get_original_cwd()}/outputs/plots/{dataset.lower()}_alpha_deltaT_{init}_density_{density}.pdf",
    #     dpi=150,
    # )
    plt.show()


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(
        f"{cfg.wandb.entity}/{cfg.wandb.project}", filters={"state": "finished"}
    )

    df = get_stats(runs)

    # Set longer length
    pd.options.display.max_rows = 150
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    df.to_csv(
        f"{hydra.utils.get_original_cwd()}/outputs/csv/{cfg.dataset.name.lower()}_alpha_deltaT.csv"
    )

    # Plot it
    alpha_deltaT_plot(
        df, init_ll=["ERK"], density_ll=[0.1, 0.2, 0.5],
    )


if __name__ == "__main__":
    main()
