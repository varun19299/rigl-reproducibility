import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os

from sparselearning.core import Masking
from sparselearning.models import registry as model_registry
from sparselearning.funcs.decay import registry as decay_registry

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
    SmoothenValue,
)

import wandb


def train(
    model: "nn.Module",
    mask: "Masking",
    train_loader: "DataLoader",
    optimizer: "optim",
    lr_scheduler: "lr_scheduler",
    global_step: int,
    epoch: int,
    device: torch.device,
    mixed_precision_scalar: "GradScaler" = None,
    log_interval: int = 100,
    use_wandb: bool = False,
    masking_apply_when: str = "epoch_end",
    masking_interval: int = 1,
    masking_end_when: int = -1,
) -> "Union[float,int]":
    assert masking_apply_when in ["step_end", "epoch_end"]
    model.train()
    _mask_update_counter = 0
    _loss_collector = SmoothenValue()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)

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
            # L2 Regularization

            # Exp avg collection
            _loss_collector.add_value(loss.item())

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if (
            mask
            and masking_apply_when == "step_end"
            and (global_step < masking_end_when)
            and (global_step % masking_interval) == 0
        ):
            mask.update_connections()
            _mask_update_counter += 1
        else:
            if mixed_precision_scalar:
                mixed_precision_scalar.step(stepper)
                mixed_precision_scalar.update()
            else:
                stepper.step()

        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)
        global_step += 1

        if batch_idx % log_interval == 0:
            msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f}"
            pbar.set_description(msg)
            if use_wandb:
                log_dict = {"train_loss": loss, "lr": lr_scheduler.get_lr()[0]}
                if mask:
                    density = mask.stats.total_density
                    log_dict = {
                        **log_dict,
                        "prune_rate": mask.prune_rate,
                        "density": density,
                    }
                wandb.log(
                    log_dict, step=global_step,
                )

    density = mask.stats.total_density if mask else 1.0
    msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f} Prune Rate {mask.prune_rate if mask else 0:.5f} Density {density:.5f}"
    logging.info(msg)

    return _loss_collector.smooth, global_step


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    global_step: int,
    epoch: int,
    device: torch.device,
    is_test_set: bool = False,
    use_wandb: bool = False,
) -> "Union[float, float]":
    model.eval()

    loss = 0
    correct = 0
    n = 0
    pbar = tqdm(total=len(loader), dynamic_ncols=True)

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

            pbar.update(1)

    accuracy = correct / n
    loss /= len(loader)

    val_or_test = "val" if not is_test_set else "test"
    msg = f"{val_or_test.capitalize()} Epoch {epoch} Iters {global_step} {val_or_test} loss {loss:.6f} accuracy {accuracy:.4f}"
    pbar.set_description(msg)
    logging.info(msg)

    # Log loss, accuracy
    if use_wandb:
        wandb.log({f"{val_or_test}_loss": loss}, step=global_step)
        wandb.log({f"{val_or_test}_accuracy": accuracy}, step=global_step)

    return loss, accuracy


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0
    model = model_class(
        *model_args, cfg.save_features, cfg.benchmark, _small_density
    ).to(device)

    # wandb
    if cfg.use_wandb:
        with open(cfg.wandb_api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read()

        _density = (
            cfg.masking.final_density
            if cfg.masking.name == "Pruning"
            else cfg.masking.density
        )
        wandb.init(
            entity="ml-reprod-2020",
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.project,
            name=f"{cfg.dataset.name}_{cfg.exp_name}_density_{_density}_alpha_{cfg.masking.prune_rate}_deltaT_{cfg.masking.interval}",
            reinit=True,
            save_code=True,
        )
        wandb.watch(model)

    # Training multiplier
    cfg.optimizer.decay_frequency *= cfg.optimizer.training_multiplier
    cfg.optimizer.decay_frequency = int(cfg.optimizer.decay_frequency)

    cfg.optimizer.epochs *= cfg.optimizer.training_multiplier
    cfg.optimizer.epochs = int(cfg.optimizer.epochs)

    cfg.masking.end_when *= cfg.optimizer.training_multiplier
    cfg.masking.end_when = int(cfg.masking.end_when)

    if cfg.masking.apply_when == "step_end":
        cfg.masking.interval *= cfg.optimizer.training_multiplier
        cfg.masking.interval = int(cfg.masking.interval)

    # Setup optimizers, lr schedulers
    optimizer, lr_scheduler = get_optimizer(model, **cfg.optimizer)

    # Mixed Precision
    mixed_precision_scalar = GradScaler() if cfg.mixed_precision else None

    # Setup mask
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "T_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "T_max": max_iter,
                "T_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = Masking(
            optimizer,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )
        # Support for lottery mask
        lottery_mask_path = cfg.masking.get("lottery_mask_path", None)
        mask.add_module(model, lottery_mask_path)

    # Load from checkpoint
    model, optimizer, mask, step, start_epoch, best_val_loss = load_weights(
        model, optimizer, mask, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    )

    # Train model
    epoch = None
    for epoch in range(start_epoch, cfg.optimizer.epochs):
        # step here is training iters not global steps
        _masking_args = {}
        if mask:
            _masking_args = {
                "masking_apply_when": cfg.masking.apply_when,
                "masking_interval": cfg.masking.interval,
                "masking_end_when": cfg.masking.end_when,
            }

        _, step = train(
            model,
            mask,
            train_loader,
            optimizer,
            lr_scheduler,
            step,
            epoch + 1,
            device,
            mixed_precision_scalar,
            log_interval=cfg.log_interval,
            use_wandb=cfg.use_wandb,
            **_masking_args,
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, step, epoch + 1, device, use_wandb=cfg.use_wandb,
        )

        # Save weights
        if (epoch + 1 == cfg.optimizer.epochs) or (
            (epoch + 1) % cfg.ckpt_interval == 0
        ):
            if val_loss < best_val_loss:
                is_min = True
                best_val_loss = val_loss
            else:
                is_min = False

            save_weights(
                model,
                optimizer,
                mask,
                val_loss,
                step,
                epoch + 1,
                ckpt_dir=cfg.ckpt_dir,
                is_min=is_min,
            )

        # Apply mask
        if (
            mask
            and cfg.masking.apply_when == "epoch_end"
            and epoch < cfg.masking.end_when
        ):
            if epoch % cfg.masking.interval == 0:
                mask.update_connections()

    if not epoch:
        # Run val anyway
        epoch = cfg.optimizer.epochs - 1
        val_loss, val_accuracy = evaluate(
            model, val_loader, step, epoch + 1, device, use_wandb=cfg.use_wandb,
        )

    evaluate(
        model,
        test_loader,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.use_wandb,
    )

    return val_accuracy


if __name__ == "__main__":
    main()
