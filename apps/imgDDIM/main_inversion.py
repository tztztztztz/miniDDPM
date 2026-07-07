import os
import sys

import torch

_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_dirname, "../imgDDPM"))

from ddim import build_ddim_schedule, inversion, p_sample
from ddpm import build_alpha, build_variance_schedule
from mnist import load_mnist
from unet import NaiveUnet
from utils import save_image_grid


T = 1000
NUM_STEPS = 100
IMG_SIZE = 28
NUM_IMAGES = 16


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def denoise(model, x_t, steps, alphas_bar):
    batch_size = x_t.shape[0]
    device = x_t.device
    reverse_steps = steps[::-1]

    for i in range(len(reverse_steps) - 1):
        t = torch.full((batch_size,), reverse_steps[i], dtype=torch.long, device=device)
        prev_t = torch.full(
            (batch_size,), reverse_steps[i + 1], dtype=torch.long, device=device
        )
        x_t = p_sample(model, x_t, t, prev_t, alphas_bar)

    return x_t


def main():
    device = get_device()
    print(f"device={device}")

    betas = build_variance_schedule(device, T)
    _, alphas_bar = build_alpha(betas)
    steps = build_ddim_schedule(T, NUM_STEPS)

    model = NaiveUnet(in_channels=1, out_channels=1, n_feat=64).to(device)
    ckpt_path = os.path.join(_dirname, "../imgDDPM/outputs/ckpt.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cache_path = os.path.join(_dirname, "../imgDDPM/.cache/mnist_data_train.pt")
    loader = load_mnist(NUM_IMAGES, cache_path=cache_path, shuffle=False)
    x_0 = next(iter(loader))[0].to(device)

    with torch.no_grad():
        x_t = inversion(model, x_0, steps, alphas_bar)
        reconstructed = denoise(model, x_t, steps, alphas_bar)

    comparison = torch.cat([x_0, reconstructed.clamp(-1.0, 1.0)], dim=0)
    save_image_grid(
        comparison,
        os.path.join(_dirname, f"outputs/ddim_inversion_{NUM_STEPS}_compare.png"),
        image_size=IMG_SIZE,
        nrow=NUM_IMAGES,
    )


if __name__ == "__main__":
    main()
