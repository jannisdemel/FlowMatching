# cfm/train.py
import torch
from torch import optim

from .models import MLPVectorField
from .losses import cfm_loss


def train_cfm(
    dataset,
    device="cpu",
    num_steps=10_000,
    batch_size=256,
    lr=1e-3,
    viz_callback=None,          # unchanged
    viz_schedule=None,          # NEW: function(step:int) -> bool
):
    """
    Unconditional flow matching training.
    If viz_schedule is provided, viz_callback(step, model) is called
    whenever viz_schedule(step) returns True.
    """
    device = torch.device(device)

    x_dim = 2
    model = MLPVectorField(x_dim=x_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for step in range(num_steps):
        x_t, t, target_v = dataset.sample_cfm_batch(batch_size)

        opt.zero_grad(set_to_none=True)
        loss = cfm_loss(model, x_t, t, target_v)
        loss.backward()
        opt.step()

        global_step = step + 1

        if global_step % 500 == 0:
            print(f"{global_step:5d} / {num_steps}  loss = {loss.item():.4f}")

        if (
            viz_callback is not None
            and viz_schedule is not None
            and viz_schedule(global_step)
        ):
            viz_callback(global_step, model)

    return model