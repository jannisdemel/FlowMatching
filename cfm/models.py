# cfm/models.py
import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Embed scalar time t in [0,1] into a higher-dimensional vector.

    t: (B,) or (B,1)
    output: (B, emb_dim)
    """

    def __init__(self, num_frequencies: int = 8, emb_dim: int = 32):
        super().__init__()
        self.num_frequencies = num_frequencies

        # frequencies like in positional encodings
        freqs = torch.pow(2.0, torch.arange(num_frequencies))
        self.register_buffer("freqs", freqs)  # (F,)

        # project [sin, cos, t] into emb_dim
        in_dim = 2 * num_frequencies + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        # t: (B,)
        t = t.unsqueeze(-1)  # (B,1)
        angles = t * self.freqs[None, :] * 2 * math.pi  # (B,F)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        feats = torch.cat([sin, cos, t], dim=-1)  # (B, 2F+1)

        return self.proj(feats)  # (B, emb_dim)


class MLPVectorField(nn.Module):
    """
    v_theta(x, t) with a sinusoidal time embedding.

    x: (B, x_dim)
    t: (B,) or (B,1)
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 32,
        time_num_frequencies: int = 8,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.time_emb = SinusoidalTimeEmbedding(
            num_frequencies=time_num_frequencies,
            emb_dim=time_emb_dim,
        )

        in_dim = x_dim + time_emb_dim  # x + time embedding

        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, x_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)           # (B, time_emb_dim)
        h = torch.cat([x, t_emb], dim=-1)  # (B, x_dim + time_emb_dim)
        return self.net(h)