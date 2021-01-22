import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig

line_alpha = 0.75

# Matplotlib font sizes
TINY_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc("axes", grid=True)
plt.rc("lines", linewidth=3)

MAX_SAMPLES = 1000000


def _export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def _get_steps_and_col(df, col, max_steps=None):
    steps = np.array(df["_step"])
    vals = np.array(df[col])
    ii = np.where(~np.isnan(np.array(vals)))
    steps = steps[ii]
    vals = vals[ii]
    if max_steps:
        jj = np.where(steps < max_steps)
        steps = steps[jj]
        vals = vals[jj]
    return steps, vals


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    # Authenticate API
    with open(cfg.wandb.api_key) as f:
        os.environ["WANDB_API_KEY"] = f.read()

    api = wandb.Api()

    riglsg, riglsm = list(
        api.runs(
            f"{cfg.wandb.entity}/{cfg.wandb.project}",
            filters={
                "state": "finished",
                "config.seed": 3,
                "config.masking.density": 0.2,
                "config.masking.sparse_init": "random",
                "config.masking.name": "RigL",
            },
        )
    )

    random_name = "Random"
    erk_name = "ERK"
    sg_name = "Sparse Grad"
    sm_name = "Sparse Mmt"

    rigl_random_flops = 0.2
    rigl_erk_flops = 0.38

    flop_col = "Avg Inference FLOPs"
    sg_history = riglsg.history(samples=MAX_SAMPLES)
    sm_history = riglsm.history(samples=MAX_SAMPLES)

    default_mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    names = [random_name, erk_name, sg_name, sm_name]

    COLORS = {}
    for i, k in enumerate(names):
        COLORS[k] = default_mpl_cycle[i]

    plt.figure(figsize=(5, 5))

    plt.axhline(
        rigl_random_flops,
        label=random_name,
        alpha=line_alpha,
        color=COLORS[random_name],
    )
    plt.axhline(
        rigl_erk_flops, label=erk_name, alpha=line_alpha, color=COLORS[erk_name]
    )
    plt.plot(
        *_get_steps_and_col(sg_history, flop_col, max_steps=3.5e4),
        label=sg_name,
        alpha=line_alpha,
        color=COLORS[sg_name],
    )
    plt.plot(
        *_get_steps_and_col(sm_history, flop_col, max_steps=3.5e4),
        label=sm_name,
        alpha=line_alpha,
        color=COLORS[sm_name],
    )

    plt.legend(loc="upper right")

    plt.grid()
    plt.xlabel("Train Step")
    plt.ylabel("Forward FLOPs")
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        f"{hydra.utils.get_original_cwd()}/outputs/plots/{cfg.wandb.project}_redist_inference_flops.pdf",
        dpi=150,
    )

    plt.show()


if __name__ == "__main__":
    main()
