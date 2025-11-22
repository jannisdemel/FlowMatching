# cfm/data.py
import math
import torch


def sample_two_moons(batch_size, noise=0.1, device="cpu"):
    """
    Simple 2D two-moons distribution (roughly like sklearn.make_moons).
    """
    device = torch.device(device)
    n = batch_size

    # sample angles in [0, pi]
    theta = torch.rand(n, device=device) * math.pi

    # first moon
    x1 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    # second moon
    x2 = torch.stack(
        [1.0 - torch.cos(theta), -torch.sin(theta) + 0.5],
        dim=-1,
    )

    # choose which moon
    mask = (torch.rand(n, device=device) < 0.5).float().unsqueeze(-1)
    x = mask * x1 + (1.0 - mask) * x2

    # add noise
    x = x + noise * torch.randn_like(x)
    return x


class TwoMoonsToGaussians2D:
    """
    Base: 2-moons distribution (x0)
    Target: mixture of K=8 Gaussians on a circle (x1)
    """

    def __init__(
        self,
        num_gaussians=8,
        radius=4.0,
        target_std=0.3,
        base_noise=0.1,
        device="cpu",
    ):
        self.num_gaussians = num_gaussians
        self.radius = radius
        self.target_std = target_std
        self.base_noise = base_noise
        self.device = torch.device(device)

        # Gaussian means on a circle
        angles = torch.linspace(0, 2 * math.pi, steps=num_gaussians + 1)[:-1]
        self.means = torch.stack(
            [radius * torch.cos(angles), radius * torch.sin(angles)],
            dim=-1,
        ).to(self.device)  # (K, 2)

    def sample_base(self, batch_size):
        """
        x0 ~ 2-moons distribution.
        """
        return sample_two_moons(
            batch_size=batch_size,
            noise=self.base_noise,
            device=self.device,
        )

    def sample_target(self, batch_size):
        """
        x1 ~ mixture of num_gaussians 2D Gaussians.
        """
        B = batch_size
        device = self.device

        comp_ids = torch.randint(
            0,
            self.num_gaussians,
            (B,),
            device=device,
        )  # (B,)
        means = self.means[comp_ids]  # (B, 2)
        noise = self.target_std * torch.randn(B, 2, device=device)
        return means + noise

    def sample_cfm_batch(self, batch_size):
        """
        Returns:
            x_t:      (B, 2)
            t:        (B,)
            target_v: (B, 2) = x1 - x0
        """
        B = batch_size
        x0 = self.sample_base(B)
        x1 = self.sample_target(B)

        t = torch.rand(B, device=self.device)   # (B,)
        tt = t.unsqueeze(-1)                    # (B,1)
        x_t = (1.0 - tt) * x0 + tt * x1         # linear path

        target_v = x1 - x0
        return x_t, t, target_v