import math
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ---------- Simple dataset holding pre-sampled pairs ----------

class PairDataset(Dataset):
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        x0, x1: [N, 2]
        """
        assert x0.shape == x1.shape, "x0 and x1 must have the same shape"
        self.x0 = x0
        self.x1 = x1

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]


# ---------- Distribution samplers ----------

def sample_gaussian(n: int, cfg: Dict[str, Any], device=None) -> torch.Tensor:
    """
    Single 2D Gaussian.
    cfg:
      mean: [2] list or tuple
      std: float (assume isotropic)
    """
    mean = torch.tensor(cfg.get("mean", [0.0, 0.0]), dtype=torch.float32, device=device)
    std = float(cfg.get("std", 1.0))
    return mean + std * torch.randn(n, 2, device=device)


def sample_two_moons(n: int, cfg: Dict[str, Any], device=None) -> torch.Tensor:
    """
    Two moons in 2D (rough sklearn-like).
    cfg:
      noise: float
      radius: float
      distance: float   # vertical separation between moons
    """
    noise = float(cfg.get("noise", 0.1))
    radius = float(cfg.get("radius", 1.0))
    distance = float(cfg.get("distance", 0.5))

    n1 = n // 2
    n2 = n - n1

    # first moon (upper)
    theta1 = torch.rand(n1, device=device) * math.pi  # [0, π]
    x1 = torch.stack(
        [
            radius * torch.cos(theta1),
            radius * torch.sin(theta1),
        ],
        dim=1,
    )

    # second moon (lower, shifted)
    theta2 = torch.rand(n2, device=device) * math.pi  # [0, π]
    x2 = torch.stack(
        [
            radius * torch.cos(theta2) + radius,
            -radius * torch.sin(theta2) - distance,
        ],
        dim=1,
    )

    x = torch.cat([x1, x2], dim=0)
    x = x + noise * torch.randn_like(x)
    return x


def sample_eight_gaussians(n: int, cfg: Dict[str, Any], device=None) -> torch.Tensor:
    """
    8 Gaussians on a circle.
    cfg:
      radius: float
      std: float
    """
    radius = float(cfg.get("radius", 2.0))
    std = float(cfg.get("std", 0.1))

    centers = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        centers.append((radius * math.cos(angle), radius * math.sin(angle)))
    centers = torch.tensor(centers, dtype=torch.float32, device=device)  # [8, 2]

    # roughly equal samples per center
    idx = torch.randint(0, 8, (n,), device=device)  # [n]
    base = centers[idx]  # [n, 2]
    return base + std * torch.randn(n, 2, device=device)


def sample_from_cfg(n: int, cfg: Dict[str, Any], device=None) -> torch.Tensor:
    """
    cfg must have key 'name' in {'gaussian', 'two_moons', 'eight_gaussians'}.
    """
    name = cfg.get("name", "gaussian")
    if name == "gaussian":
        return sample_gaussian(n, cfg, device=device)
    elif name == "two_moons":
        return sample_two_moons(n, cfg, device=device)
    elif name == "eight_gaussians":
        return sample_eight_gaussians(n, cfg, device=device)
    else:
        raise ValueError(f"Unknown distribution name: {name}")


# ---------- Lightning DataModule ----------

class RectifiedFlow2DDataModule(pl.LightningDataModule):
    """
    Provides (x0, x1) pairs in R^2 for rectified flows
    from a source distribution to a target distribution.
    """

    def __init__(
        self,
        batch_size: int = 512,
        n_train: int = 50_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        source_dist: Optional[Dict[str, Any]] = None,
        target_dist: Optional[Dict[str, Any]] = None,
        num_workers: int = 0,
    ):
        """
        Args:
            batch_size: batch size
            n_train, n_val, n_test: dataset sizes
            source_dist: dict with at least 'name' key
            target_dist: dict with at least 'name' key
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.source_dist = source_dist or {"name": "gaussian"}
        self.target_dist = target_dist or {"name": "two_moons"}
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        device = self._get_device_for_data()

        # train
        x0_train = sample_from_cfg(self.n_train, self.source_dist, device=device)
        x1_train = sample_from_cfg(self.n_train, self.target_dist, device=device)
        self.train_dataset = PairDataset(x0_train, x1_train)

        # val
        x0_val = sample_from_cfg(self.n_val, self.source_dist, device=device)
        x1_val = sample_from_cfg(self.n_val, self.target_dist, device=device)
        self.val_dataset = PairDataset(x0_val, x1_val)

        # test
        x0_test = sample_from_cfg(self.n_test, self.source_dist, device=device)
        x1_test = sample_from_cfg(self.n_test, self.target_dist, device=device)
        self.test_dataset = PairDataset(x0_test, x1_test)

    def _get_device_for_data(self):
        # simple: put data on CPU, the model will move it to GPU
        return torch.device("cpu")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )