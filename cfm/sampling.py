# cfm/sampling.py
import torch


def euler_sample(
    model,
    base_sampler,
    num_samples: int,
    num_steps: int = 100,
    device: str = "cpu",
):
    """
    Integrate dx/dt = v_theta(x, t) from t=0 to t=1 starting from x0 ~ base_sampler.

    base_sampler: function (batch_size) -> (B, 2) tensor on correct device
    """
    device = torch.device(device)

    x = base_sampler(num_samples).to(device)  # (B, 2)
    dt = 1.0 / num_steps
    B = x.shape[0]

    for i in range(num_steps):
        t = torch.full((B,), (i + 0.5) * dt, device=device)  # (B,)
        v = model(x, t)                                      # (B, 2)
        x = x + dt * v

    return x  # (B, 2)