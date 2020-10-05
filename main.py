import hydra
from omegaconf import DictConfig, OmegaConf
import os

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import registry as model_registry


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

from utils.train_helper import (
    get_dataloaders,
    get_optimizer,
    load_weights,
    save_weights,
)

import wandb


def train(
    model: "nn.Module",
    mask: "Masking",
    train_loader: "DataLoader",
    optimizer: "optim",
    lr_scheduler: "lr_scheduler",
    epoch: int,
    device: torch.device,
    pbar: "tqdm",
    mixed_precision_scalar: "GradScaler" = None,
    log_interval: int = 10,
    use_wandb: bool = False,
):
    model.train()
    pbar.write(f"Starting Epoch {epoch}...")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if mixed_precision_scalar:
            with autocast():
                output = model(data)
                loss = F.nll_loss(output, target)

            mixed_precision_scalar.scale(loss).backward()

        else:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if mixed_precision_scalar:
            mixed_precision_scalar.step(stepper)
            mixed_precision_scalar.update()
        else:
            stepper.step()

        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)

        if batch_idx % log_interval == 0:
            pbar.set_description(f"Train Epoch {epoch} train loss {loss.item():.6f}")
            step = (
                epoch * len(train_loader.dataset)
                + batch_idx * train_loader.batch_size
                + 1
            )
            if use_wandb:
                wandb.log({"train_loss": loss}, step=step)

    pbar.refresh()
    pbar.reset()


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    epoch: int,
    device: torch.device,
    pbar: "tqdm",
    is_test_set: bool = False,
) -> "Union[float, float]":
    model.eval()

    loss = 0
    correct = 0
    n = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            model.t = target
            output = model(data)
            loss += F.nll_loss(output, target).item()  # sum up batch loss

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            n += target.shape[0]

    accuracy = correct / n

    val_or_test = "val" if not is_test_set else "test"
    pbar.set_description(
        f"{val_or_test.capitalize()} Epoch {epoch} {val_or_test} loss {loss.item():.6f} accuracy {accuracy:.4f}"
    )
    pbar.refresh()
    pbar.reset()

    return loss, accuracy


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    breakpoint()

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(**cfg.dataset)

    # Select model
    assert (
        cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    model = model_class(*(model_args + [cfg.save_features, cfg.benchmark])).to(device)

    # wandb
    if cfg.use_wandb:
        with open(cfg.wandb_api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read()

        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project="Sparse Network Training",
            name=f"{cfg.exp_name}",
            reinit=True,
            save_code=True,
        )
        wandb.watch(model)

    # Setup optimizers, lr schedulers
    optimizer, lr_scheduler = get_optimizer(model, **cfg.optimizer)

    # Mixed Precision
    mixed_precision_scalar = GradScaler() if cfg.mixed_precision else None

    # Load from checkpoint
    model, optimizer, step, start_epoch, best_val_loss = load_weights(
        model, optimizer, ckpt_dir=cfg.ckpt_dir
    )

    # Setup mask
    mask = None
    if not cfg.masking.dense:
        if cfg.masking.decay_schedule == "cosine":
            decay = CosineDecay(
                cfg.masking.prune_rate, len(train_loader) * (cfg.optimizer.epochs)
            )
        elif cfg.masking.decay_schedule == "linear":
            decay = LinearDecay(
                cfg.masking.prune_rate, len(train_loader) * (cfg.optimizer.epochs)
            )
        mask = Masking(
            optimizer,
            decay,
            prune_rate=cfg.masking.prune_rate,
            prune_mode=cfg.masking.prune,
            growth_mode=cfg.masking.growth,
            redistribution_mode=cfg.masking.redistribution,
            verbose=cfg.masking.verbose,
        )
        mask.add_module(model, density=False)

    # pbar
    train_pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    val_pbar = tqdm(total=len(val_loader), dynamic_ncols=True)
    test_pbar = tqdm(total=len(test_loader), dynamic_ncols=True)

    # Train model
    for epoch in range(start_epoch, cfg.optimizer.epochs):
        train(
            model,
            mask,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch + 1,
            device,
            train_pbar,
            mixed_precision_scalar,
            log_interval=cfg.log_interval,
            use_wandb=cfg.use_wanb,
        )

        val_loss, val_accuracy = evaluate(model, val_loader, epoch, device, val_pbar,)

        # Validation loss, accuracy
        if cfg.use_wandb:
            step = epoch * len(train_loader.dataset) + 1
            wandb.log({"val_loss": val_loss}, step=step)
            wandb.log({"val_accuracy": val_accuracy}, step=step)

        # Save weights
        if (epoch + 1) % cfg.ckpt_interval == 0:
            step = epoch * len(train_loader.dataset) + 1
            if val_loss < best_val_loss:
                is_min = True
                best_val_loss = val_loss

            save_weights(
                model,
                optimizer,
                val_loss,
                step,
                epoch,
                ckpt_dir=cfg.ckpt_dir,
                is_min=is_min,
            )

        # Apply mask
        if (
            mask
            and cfg.masking.apply_when == "epoch_end"
            and epoch < cfg.masking.epochs
        ):
            mask.update_connections()

    test_loss, test_accuracy = evaluate(model, test_loader, epoch, device, test_pbar)

    # Validation loss, accuracy
    if cfg.use_wandb:
        step = epoch * len(train_loader.dataset) + 1
        wandb.log({"test_loss": test_loss}, step=step)
        wandb.log({"test_accuracy": test_accuracy}, step=step)


if __name__ == "__main__":
    main()
