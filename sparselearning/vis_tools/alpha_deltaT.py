"""
Run as:

python sparselearning/vis_tools/alpha_deltaT.py wandb.project="cifar10 optuna multiseed" dataset=CIFAR10
"""
import hydra
import itertools
import logging
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import os
import pandas as pd
import seaborn as sns
import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *

# Seaborn
sns.set_theme()

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
        "Val Acc (seed 0)",
        "Val Acc (seed 1)",
        "Val Acc (seed 2)",
        "Test Acc (seed 0)",
        "Test Acc (seed 1)",
        "Test Acc (seed 2)",
        "Val Acc",
        "Test Acc",
    ]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    runs_dict = {}
    for run in runs:
        if run.name not in runs_dict:
            runs_dict[run.name] = [run]
        else:
            runs_dict[run.name].append(run)

    for e, run_name in enumerate(runs_dict.keys()):
        run_ll = runs_dict[run_name]
        masking = run_ll[0].config["masking"]["name"]
        init = (
            "ERK"
            if run_ll[0].config["masking"]["sparse_init"] == "erdos-renyi-kernel"
            else "Random"
        )
        density = run_ll[0].config["masking"]["density"]
        alpha = run_ll[0].config["masking"]["prune_rate"]
        deltaT = run_ll[0].config["masking"]["interval"]

        df.loc[e] = [masking, init, density, alpha, deltaT, *([None, None] * 4)]

        for run in run_ll:
            val_accuracy = run.summary.val_accuracy * 100
            test_accuracy = run.summary.test_accuracy * 100
            seed = run.config["seed"]
            df.loc[e, f"Val Acc (seed {seed})"] = val_accuracy
            df.loc[e, f"Test Acc (seed {seed})"] = test_accuracy

        df.loc[e, "Val Acc"] = (
            df.loc[e][[f"Val Acc (seed {i})" for i in range(3)]].mean().astype(float)
        )
        df.loc[e, "Test Acc"] = (
            df.loc[e][[f"Test Acc (seed {i})" for i in range(3)]].mean().astype(float)
        )

    df = df.sort_values(by=["Method", "Init", "Density", "alpha", "Delta T"])
    df = df.dropna()

    if reorder:
        df = df.reset_index(drop=True)
    return df


def alpha_deltaT_plot(
    df: pd.DataFrame,
    dataset: str = "CIFAR10",
    init_ll: "List[str]" = ["ERK", "Random"],
    density_ll=[0.1, 0.2, 0.5],
):
    for e, (init, density) in enumerate(itertools.product(init_ll, density_ll)):
        sub_df = df.loc[(df["Init"] == init) & (df["Density"] == density)]
        row = sub_df.loc[sub_df["Test Acc"].astype(float).idxmax()]

        print(row)
        print("\n")

        alpha_ll = sub_df["alpha"].astype(float)
        deltaT_ll = sub_df["Delta T"].astype(float)
        test_accuracy_ll = sub_df["Test Acc"].astype(float).to_numpy()

        plt.tricontourf(alpha_ll, deltaT_ll, test_accuracy_ll, levels=10, cmap="plasma")
        # plt.tricontour(alpha_ll, deltaT_ll, test_accuracy_ll, levels=10)

        plt.plot(0.3, 100, "ko", markersize=10)

        # plt.plot(row["alpha"], row["Delta T"], "^", color="#F5F5F5", markersize=8)
        plt.plot(row["alpha"], row["Delta T"], "^", color="black", markersize=10)

        cbar = plt.colorbar()
        cbar.set_label("Accuracy (Test)")

        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\Delta T$")

        plt.ylim(50, 1000)

        plt.savefig(
            f"{hydra.utils.get_original_cwd()}/outputs/plots/{dataset.lower()}_alpha_deltaT_{init}_density_{density}.pdf",
            dpi=150,
        )

        plt.show()


@hydra.main(config_name="config", config_path="../../conf")
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
        df, init_ll=["ERK", "Random"], density_ll=[0.1, 0.2, 0.5],
    )


if __name__ == "__main__":
    main()
