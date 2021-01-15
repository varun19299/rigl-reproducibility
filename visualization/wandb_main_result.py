"""
Run as:

python visualization/wandb_main_results.py wandb.project=cifar10 dataset=CIFAR10

python visualization/wandb_main_results.py wandb.project=cifar100 dataset=CIFAR100
"""
import os
from typing import TYPE_CHECKING

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from visualization.main_results import get_stats_table

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


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
            "SNFS",
            "SET",
            "Small_Dense",
            "Dense",
            "Static",
            "Pruning",
        ],
        init_ll=["Random", None],
        suffix_ll=[None],
        density_ll=[0.05, 0.1, 0.2, 0.5, 1],
        dataset_ll=[cfg.dataset.name],
        correct_SET=cfg.dataset.name == "CIFAR10",
    )

    # Set longer length
    pd.options.display.max_rows = 150

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df)

    config = {"dataset": cfg.dataset.name}

    for method in df["Method"].unique():
        wandb.init(
            entity=cfg.wandb.entity,
            config=config,
            project="report",
            name=method,
            reinit=True,
        )
        sub_df = df.loc[df["Method"] == method][["Density", "Mean Acc"]]
        table = wandb.Table(dataframe=sub_df)

        wandb.log(
            {
                "Random Initialization": wandb.plot.line(
                    table, "Density", "Mean Acc", title="Test Accuracy vs Sparsity"
                )
            },
            step=0,
        )
        # sub_df_seeds = df.loc[df["Method"] == method][
        #     ["Density", "Acc seed 0", "Acc seed 1", "Acc seed 2"]
        # ]


if __name__ == "__main__":
    main()
