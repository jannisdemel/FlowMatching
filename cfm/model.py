# mnist/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from cfm.data import sample_from_cfg



class FullyConnectedMLP(nn.Module):
    """Fully connected network R^in_dim -> R^out_dim with configurable hidden width."""

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 128,
        hidden_dim: int = 128,
        n_hidden_layers: int = 0,
    ):
        """
        Args:
            in_dim: input dimension
            out_dim: output dimension
            hidden_dim: width of hidden layers
            n_hidden_layers: number of hidden layers
        """
        super().__init__()
        self.out_dim = out_dim

        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))

        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)




class MLP_with_time(pl.LightningModule):
    def __init__(
        self,
        data_dim: int = 2,
        time_embedding: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hidden_dim: int = 0,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["time_embedding"])

        if time_embedding is None:
            raise ValueError("time_embedding must be provided and map R^1 -> R^emb_dim")

        self.time_embedding = time_embedding  # should have attribute .out_dim 

        if hidden_dim == 0:
            self.hidden_dim = 3 * data_dim
        else:
            self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # input = x_t (data_dim) + time embedding (emb_dim)
        emb_dim = getattr(self.time_embedding, "out_dim", None)
        if emb_dim is None:
            raise ValueError("time_embedding must define an attribute 'out_dim' giving its output dimension")

        self.model = FullyConnectedMLP(
            in_dim=data_dim + emb_dim,
            out_dim=data_dim,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, data_dim]
        t:   [B] or [B,1]
        returns v_theta(x_t, t): [B, data_dim]
        """
        t_emb = self.time_embedding(t.unsqueeze(-1))         # [B, emb_dim]
        # concat along feature dimension
        h = torch.cat([x_t, t_emb], dim=-1)    # [B, data_dim + emb_dim]
        return self.model(h)                  # [B, data_dim]


    def training_step(self, batch, batch_idx):
        """
        Rectified-flow loss between two arbitrary distributions.

        batch: (x0, x1)
            x0: [B, data_dim]  (source samples)
            x1: [B, data_dim]  (target samples)
        """
        x0, x1 = batch
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        B = x0.size(0)
        t = torch.rand(B, device=self.device)          # [B]
        t_b = t.unsqueeze(-1)                          # [B,1]

        # straight-line interpolation
        x_t = (1.0 - t_b) * x0 + t_b * x1              # [B, data_dim]
        v_target = x1 - x0                             # [B, data_dim] (constant along the path)

        v_pred = self(x_t, t)                          # [B, data_dim]

        loss = F.mse_loss(v_pred, v_target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1 = batch
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        B = x0.size(0)
        t = torch.rand(B, device=self.device)
        t_b = t.unsqueeze(-1)

        x_t = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self(x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        self.log("val_loss", loss, prog_bar=True)
    
    @torch.no_grad()
    def sample_flow(
        self,
        n_samples: int,
        source_cfg: dict,
        n_steps: int = 100,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Sample from the *learned* flow: source -> target.

        Args:
            n_samples: number of samples
            source_cfg: config dict for the source distribution
            n_steps: number of Euler steps in [0,1]
        Returns:
            x_T: [n_samples, data_dim] samples approximating target dist
        """
        if device is None:
            device = self.device

        self.eval()

        # x0 ~ source distribution
        x = sample_from_cfg(n_samples, source_cfg, device=device)  # [N, data_dim]

        t0, t1 = 0.0, 1.0
        dt = (t1 - t0) / n_steps

        for k in range(n_steps):
            t_k = t0 + (k + 0.5) * dt
            t = torch.full((n_samples,), t_k, device=device)  # [N]
            v = self(x, t)                                   # [N, data_dim]
            x = x + dt * v

        return x  # approximate target samples

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


