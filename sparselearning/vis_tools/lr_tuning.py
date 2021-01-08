"""
Run as:

python sparselearning/vis_tools/lr_tuning.py wandb.project="cifar10 grid lr" dataset=CIFAR10
"""
import itertools
import logging
import os
from typing import TYPE_CHECKING

import hydra
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


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
LINE_WIDTH = 3
ALPHA = 0.9


def get_stats(
    runs,
    masking_ll: "List[str]" = ["RigL"],
    init_ll: "List[str]" = ["Random"],
    suffix_ll: "List[str]" = ["grid_lr"],
    density_ll: "List[float]" = [0.1],
    lr_ll: "List[float]" = [0.1],
    alpha_ll: "List[float]" = [0.3],
    deltaT_ll: "List[float]" = [100],
    dataset_ll: "List[str]" = ["CIFAR10"],
    reorder: bool = True,
) -> pd.DataFrame:
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
        "LR",
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

    for e, (dataset, masking, suffix, init, density, alpha, deltaT, lr) in enumerate(
        itertools.product(
            dataset_ll,
            masking_ll,
            suffix_ll,
            init_ll,
            density_ll,
            alpha_ll,
            deltaT_ll,
            lr_ll,
        )
    ):

        tags = [
            dataset,
            masking,
            init,
            suffix,
            f"density_{density}",
            f"alpha_{alpha}",
            f"deltaT_{deltaT}",
            f"lr_{lr}",
        ]
        name = "_".join([tag for tag in tags if tag])
        logging.debug(name)
        runs = runs_dict.get(name, None)

        if not runs:
            continue

        accuracy_ll = [None, None]
        for run in runs:
            if not ("test_accuracy" in run.summary):
                continue

            if not ("val_accuracy" in run.summary):
                continue

            accuracy_ll[0] = run.summary.val_accuracy * 100
            accuracy_ll[1] = run.summary.test_accuracy * 100
            break

        if suffix:
            masking = f"{masking}_{suffix}"
        df.loc[e] = [masking, init, density, alpha, deltaT, lr, *accuracy_ll]

    df = df.sort_values(by=["Method", "Init", "Density", "alpha", "Delta T", "LR"])

    if reorder:
        df = df.reset_index(drop=True)
    return df


def lr_tuning_plot(
    df: pd.DataFrame,
    dataset: str = "CIFAR10",
    init_ll: "List[str]" = ["ERK", "Random"],
    density_ll=[0.1, 0.2, 0.5],
    lr_ll: "List[float]" = [0.1],
    alpha_ll: "List[float]" = [0.3],
    deltaT_ll: "List[float]" = [100],
):
    for (init, density) in itertools.product(init_ll, density_ll):
        sub_df = df.loc[(df["Init"] == init) & (df["Density"] == density)]
        legend = []
        for (alpha, deltaT) in itertools.product(alpha_ll, deltaT_ll):
            rows = sub_df.loc[(df["alpha"] == alpha) & (df["Delta T"] == deltaT)]
            if rows.empty:
                continue

            test_acc_exists = {
                lr: not rows[rows["LR"] == lr]["Test Acc"].empty for lr in lr_ll
            }
            test_acc_ll = [
                rows[rows["LR"] == lr]["Test Acc"].iloc[0]
                for lr in lr_ll
                if test_acc_exists[lr]
            ]
            lr_ll = [lr for lr in lr_ll if test_acc_exists[lr]]

            plt.semilogx(lr_ll, test_acc_ll, linewidth=LINE_WIDTH, alpha=ALPHA)
            legend.append(rf"$\alpha=${alpha},$\Delta T=${deltaT}")

        plt.legend(legend, loc="lower left")
        plt.xticks(lr_ll, lr_ll)

        # grab a reference to the current axes
        ax = plt.gca()

        # set the xlimits to be the reverse of the current xlimits
        ax.set_xlim(ax.get_xlim()[::-1])

        plt.grid()
        plt.xlabel("Learning rate")
        plt.ylabel("Accuracy (Test)")

        plt.subplots_adjust(bottom=0.125)
        # plt.title(f"RigL {init}, $1-s=${density}")
        plt.savefig(
            f"{hydra.utils.get_original_cwd()}/outputs/plots/{dataset.lower()}_lr_tuning_{init}_density_{density}.pdf",
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
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    df = get_stats(
        runs,
        masking_ll=["RigL",],
        init_ll=["Random", "ERK"],
        density_ll=[0.1, 0.2, 0.5],
        dataset_ll=[cfg.dataset.name],
        lr_ll=[0.005, 0.01, 0.05, 0.1, 0.2],
        alpha_ll=[0.3, 0.4, 0.5],
        deltaT_ll=[100, 200, 500, 750],
    )

    # Set longer length
    pd.options.display.max_rows = 150
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    df.to_csv(
        f"{hydra.utils.get_original_cwd()}/outputs/csv/{cfg.dataset.name.lower()}_lr_tuning.csv"
    )

    # Plot it
    lr_tuning_plot(
        df,
        init_ll=["ERK", "Random"],
        density_ll=[0.1, 0.2, 0.5],
        lr_ll=[0.005, 0.01, 0.05, 0.1],
        alpha_ll=[0.3, 0.4, 0.5],
        deltaT_ll=[100, 200, 500, 750],
    )


if __name__ == "__main__":
    main()
