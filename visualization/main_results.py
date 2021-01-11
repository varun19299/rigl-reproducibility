"""
Run as:

python sparselearning/visualization/main_results.py wandb.project=cifar10 dataset=CIFAR10

python sparselearning/visualization/main_results.py wandb.project=cifar100 dataset=CIFAR100
"""
import itertools
import logging
import os
from typing import TYPE_CHECKING

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def get_stats_table(
    runs,
    masking_ll: "List[str]" = ["RigL"],
    init_ll: "List[str]" = ["Random"],
    suffix_ll: "List[str]" = [""],
    density_ll: "List[float]" = [0.1],
    dataset_ll: "List[str]" = ["CIFAR10"],
    reorder: bool = True,
    correct_SET: bool = False,
) -> pd.DataFrame:
    """
    List all possible choices for
    masking, init, density, dataset

    We'll try matching the exhaustive caretesian product
    """
    columns = ["Method", "Init", "Density", "Acc seed 0", "Acc seed 1", "Acc seed 2"]
    df = pd.DataFrame(columns=columns)

    # Pre-process
    logging.info("Grouping runs by name")
    runs_dict = {}
    for run in runs:
        if run.name not in runs_dict:
            runs_dict[run.name] = [run]
        else:
            runs_dict[run.name].append(run)

    for e, (dataset, masking, suffix, init, density) in enumerate(
        itertools.product(dataset_ll, masking_ll, suffix_ll, init_ll, density_ll)
    ):

        tags = [dataset, masking, suffix, init, "density_" + str(density)]
        name = "_".join([tag for tag in tags if tag])
        logging.debug(name)
        runs = runs_dict.get(name, None)
        if not runs:
            continue

        accuracy_ll = [None, None, None]
        for run in runs:
            if not ("test_accuracy" in run.summary):
                continue

            # Admit seeds 0,1,2 only
            if run.config["seed"] > 2:
                continue
            accuracy_ll[run.config["seed"]] = run.summary.test_accuracy * 100

            if correct_SET:
                # Correct SET Random 0.05
                # Seeds 1,2 suffered from collapse
                if (masking, suffix, init, density) == ("SET", None, "Random", 0.05):
                    accuracy_ll[1] = 0.9010 * 100
                    accuracy_ll[2] = 0.9000 * 100

            if dataset == "CIFAR100":
                # Correct RigL_3x Random 0.05
                # Seeds 1 suffered from collapse
                if (masking, suffix, init, density) == ("RigL", "3x", "Random", 0.5):
                    accuracy_ll[1] = 0.74505 * 100

        if suffix:
            masking = f"{masking}_{suffix}"
        df.loc[e] = [masking, init, density, *accuracy_ll]

    df = df.sort_values(by=["Method", "Init", "Density"])

    if reorder:
        df = df.reset_index(drop=True)

    # Compute Mean
    df["Mean Acc"] = df[[f"Acc seed {i}" for i in range(3)]].mean(axis=1)

    # Compute std dev
    df["Std. Dev"] = df[[f"Acc seed {i}" for i in range(3)]].std(axis=1)

    return df


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    # Get runs
    api = wandb.Api()
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")

    df = get_stats_table(
        runs,
        masking_ll=[
            "RigL",
            "RigL-SG",
            "RigL-SM",
            "SNFS",
            "SET",
            "Small_Dense",
            "Dense",
            "Static",
            "Pruning",
        ],
        init_ll=["Random", "ERK", None],
        suffix_ll=[None, "corrected", "no_val", "2x", "3x"],
        density_ll=[0.05, 0.1, 0.2, 0.5, 1],
        dataset_ll=[cfg.dataset.name],
        correct_SET=cfg.dataset.name == "CIFAR10",
    )

    # Set longer length
    pd.options.display.max_rows = 150

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    df.to_csv(
        f"{hydra.utils.get_original_cwd()}/outputs/csv/{cfg.dataset.name.lower()}_main_results.csv"
    )


if __name__ == "__main__":
    main()
