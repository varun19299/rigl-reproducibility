from pathlib import Path
import torch
from torch.nn import functional as F

from models.wide_resnet import WideResNet
from sparselearning.core import Masking
from sparselearning.funcs.decay import CosineDecay


def save(model, optimizer, mask, step):
    state_dict = {
        "step": step,
        "model": model.state_dict(),
        "mask": mask.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_path = Path(f"/tmp/tests/test_save_{step}.pth")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(state_dict, save_path)


def load(model, optimizer, mask, step):
    save_path = Path(f"/tmp/tests/test_save_{step}.pth")
    state_dict = torch.load(save_path, map_location="cpu")

    step = state_dict["step"]
    mask.load_state_dict(state_dict["mask"])
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])

    return model, optimizer, mask, step


def test_save_load():
    """
    1. Initialise
    2. Save
    3. Load
        Assert if equal
    4. Perform optim step
    """
    # Initialise
    model = WideResNet(depth=22, widen_factor=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    decay = CosineDecay()
    mask = Masking(optimizer, decay)
    mask.add_module(model)

    step = 0

    save(model, optimizer, mask, step)
    new_model, new_optimizer, new_mask, new_step = load(model, optimizer, mask, step)

    assert new_step == step
    assert new_model == model
    assert new_mask == mask

    for step in range(5):
        dummy_input = torch.rand(1, 3, 32, 32)
        output = model(dummy_input)
        loss = F.mse_loss(output, torch.zeros_like(output))

        loss.backward()
        assert model == mask.module

        print(f"Loss {loss}")

        if step == 5:
            mask.update_connections()
        else:
            mask.step()

    save(model, optimizer, mask, step)
    new_model, new_optimizer, new_mask, new_step = load(model, optimizer, mask, step)

    assert new_step == step
    assert new_model == model
    assert new_mask.stats.total_density == mask.stats.total_density

    # Re-initialise
    model = WideResNet(depth=22, widen_factor=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    decay = CosineDecay()
    mask = Masking(optimizer, decay)
    mask.add_module(model)


if __name__ == "__main__":
    test_save_load()
