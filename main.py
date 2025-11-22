# main.py
import torch

from cfm.data import TwoMoonsToGaussians2D
from cfm.train import train_cfm
from cfm.viz import make_frame_saver, make_gif_and_cleanup


def viz_schedule(step: int) -> bool:
    """
    Screenshot schedule:
      - up to 100: every 20
      - 101..1000: every 50
      - >1000: every 300
    """
    if step <= 100:
        return step % 20 == 0
    elif step <= 1000:
        return step % 50 == 0
    else:
        return step % 300 == 0


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training unconditional Flow Matching (2-moons -> 8 Gaussians)...")

    dataset = TwoMoonsToGaussians2D(device=device)

    frames_dir = "outputs/frames"
    viz_callback = make_frame_saver(
        dataset=dataset,
        frames_dir=frames_dir,
        device=device,
        num_samples=3* 1024,
        num_steps_integrate=200,
    )

    model = train_cfm(
        dataset=dataset,
        device=device,
        num_steps=10_000,
        batch_size=256,
        lr=1e-3,
        viz_callback=viz_callback,
        viz_schedule=viz_schedule,
    )

    make_gif_and_cleanup(
        frames_dir=frames_dir,
        out_path="outputs/cfm_unconditional_training.gif",
        fps=4,
    )