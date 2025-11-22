import os
import glob
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import imageio.v2 as imageio
import torch

from cfm.data import sample_from_cfg


class SamplingGifCallback(pl.Callback):
    '''Callback to generate sampling plots during training and compile them into a GIF at the end.
    This only makes sense for 2D data, where we can visualize the samples easily.'''
    def __init__(
        self,
        n_samples: int = 2048,
        n_steps: int = 100,
        every_n_epochs: int = 1,
        fig_size=(5, 5),
        dpi=120,
        out_dir: str = "sampling_plots",
        gif_name: str = "training_samples.gif",
        fps: int = 5,
        xlim=None,   # e.g. [-3, 3]
        ylim=None,   # e.g. [-3, 3]
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.every_n_epochs = every_n_epochs
        self.fig_size = fig_size
        self.dpi = dpi
        self.out_dir = out_dir
        self.gif_name = gif_name
        self.fps = fps
        self.xlim = xlim
        self.ylim = ylim

        os.makedirs(self.out_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        dm = trainer.datamodule
        model = pl_module

        # model samples
        x_model = model.sample_flow(
            n_samples=self.n_samples,
            source_cfg=dm.source_dist,
            n_steps=self.n_steps,
            device=model.device,
        ).detach().cpu()

        # true target samples
        x_target = sample_from_cfg(
            self.n_samples,
            dm.target_dist,
            device=model.device,
        ).detach().cpu()

        # figure with fixed pixel size (figsize + dpi)
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)

        ax.scatter(x_target[:, 0], x_target[:, 1], alpha=0.3, s=5, label="target")
        ax.scatter(x_model[:, 0], x_model[:, 1], alpha=0.3, s=5, label="model")
        ax.set_aspect("equal", "box")
        ax.legend()
        ax.set_title(f"Epoch {epoch}: model vs target")

        # fixed axis limits: either user-specified or inferred from first call
        if self.xlim is None or self.ylim is None:
            all_x = torch.cat([x_target, x_model], dim=0)
            margin = 0.2
            xmin, xmax = all_x[:, 0].min().item(), all_x[:, 0].max().item()
            ymin, ymax = all_x[:, 1].min().item(), all_x[:, 1].max().item()
            self.xlim = [xmin - margin, xmax + margin]
            self.ylim = [ymin - margin, ymax + margin]

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # save WITHOUT bbox_inches="tight" → constant canvas size
        fname = os.path.join(self.out_dir, f"epoch_{epoch:04d}.png")
        fig.savefig(fname)
        plt.close(fig)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pattern = os.path.join(self.out_dir, "epoch_*.png")
        files = sorted(glob.glob(pattern))
        if not files:
            print("SamplingGifCallback: no images found, skipping GIF creation.")
            return

        images = [imageio.imread(f) for f in files]
        gif_path = os.path.join(self.out_dir, self.gif_name)
        imageio.mimsave(gif_path, images, fps=self.fps)
        print(f"SamplingGifCallback: saved GIF to {gif_path}")

        # clean up pngs
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Could not remove {f}: {e}")