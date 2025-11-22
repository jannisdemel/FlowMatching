# cfm/viz.py
import os
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .sampling import euler_sample


def _compute_target_density_grid(dataset, x_min, x_max, y_min, y_max, n_grid=200):
    """
    Compute mixture-of-Gaussians target density on a regular grid.
    Assumes dataset has attributes:
        - means: (K, 2)
        - target_std: float
    """
    device = dataset.device
    means = dataset.means.to(device)          # (K, 2)
    sigma = dataset.target_std
    var = sigma ** 2

    xs = torch.linspace(x_min, x_max, n_grid, device=device)
    ys = torch.linspace(y_min, y_max, n_grid, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")     # (n_grid, n_grid)

    # grid points as (N,2)
    grid = torch.stack([X, Y], dim=-1)               # (n_grid, n_grid, 2)
    K = means.shape[0]

    # expand for broadcasting: (n_grid, n_grid, K, 2)
    diff = grid.unsqueeze(2) - means.view(1, 1, K, 2)

    # squared distances: (n_grid, n_grid, K)
    sq = (diff ** 2).sum(dim=-1)

    # Gaussian densities per component
    norm_const = 1.0 / (2.0 * math.pi * var)
    comp_dens = norm_const * torch.exp(-0.5 * sq / var)  # (n_grid, n_grid, K)

    # mixture with equal weights
    dens = comp_dens.mean(dim=-1)                        # (n_grid, n_grid)

    return X.cpu(), Y.cpu(), dens.cpu()


def make_frame_saver(
    dataset,
    frames_dir,
    device,
    num_samples=1024,
    num_steps_integrate=200,
):
    """
    Returns viz_callback(step, model) that:
      - samples from model (using dataset.sample_base)
      - plots samples + contour of target density
      - saves to frames_dir/frame_{step:05d}.png
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Fixed plot bounds to keep all frames same size
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6

    # precompute target density grid once
    X, Y, dens = _compute_target_density_grid(
        dataset,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_grid=200,
    )

    def viz_callback(step, model):
        model.eval()
        with torch.no_grad():
            base_sampler = lambda B: dataset.sample_base(B)
            samples = euler_sample(
                model=model,
                base_sampler=base_sampler,
                num_samples=num_samples,
                num_steps=num_steps_integrate,
                device=device,
            ).detach().cpu()  # (num_samples, 2)

        # --- plotting ---
        plt.figure(figsize=(6, 6))

        # contour of target density
        # choose a few log-spaced or linear levels
        levels = 10
        plt.contour(
            X.numpy(),
            Y.numpy(),
            dens.numpy(),
            levels=levels,
            linewidths=1.0,
            alpha=0.7,
        )

        # scatter of model samples
        x_np = samples.numpy()
        plt.scatter(
            x_np[:, 0],
            x_np[:, 1],
            s=5,
            alpha=0.5,
            color="tab:blue",
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"CFM samples vs target at step {step}")
        plt.tight_layout()

        out_path = frames_dir / f"frame_{step:05d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved frame {out_path}")

    return viz_callback


def make_gif_and_cleanup(frames_dir, out_path, fps=5):
    """
    Build GIF from all frame_*.png in frames_dir, then delete them.
    """
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print("No frames found, not creating GIF.")
        return

    images = [imageio.imread(f) for f in frame_files]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, images, fps=fps)
    print(f"Saved GIF to {out_path}")

    # cleanup
    for f in frame_files:
        os.remove(f)
    print(f"Deleted {len(frame_files)} frame files.")