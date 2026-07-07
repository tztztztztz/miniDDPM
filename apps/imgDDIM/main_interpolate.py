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
NUM_PAIRS = 8
NUM_INTERPOLATIONS = 16


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


def interpolate_pairs(latents):
    """
    Use spherical linear interpolation (slerp) to interpolate between two latents.
    Because most of the mass of Gaussian distribution is on the surface of the sphere, 
    so we use slerp to interpolate between two latents.
    Reference: https://en.wikipedia.org/wiki/Slerp
    """
    pairs = latents.view(NUM_PAIRS, 2, *latents.shape[1:])
    x_0 = pairs[:, 0].flatten(1)
    x_1 = pairs[:, 1].flatten(1)
    weights = torch.linspace(0.0, 1.0, steps=NUM_INTERPOLATIONS, device=latents.device)

    x_0_norm = x_0 / x_0.norm(dim=1, keepdim=True).clamp_min(1e-8)
    x_1_norm = x_1 / x_1.norm(dim=1, keepdim=True).clamp_min(1e-8)
    dot = (x_0_norm * x_1_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    weights = weights.view(1, NUM_INTERPOLATIONS, 1)
    x_0 = x_0[:, None]
    x_1 = x_1[:, None]
    omega = omega[:, None]
    sin_omega = sin_omega[:, None]

    slerped = (
        torch.sin((1.0 - weights) * omega) / sin_omega * x_0
        + torch.sin(weights * omega) / sin_omega * x_1
    )
    lerped = (1.0 - weights) * x_0 + weights * x_1
    interpolated = torch.where(sin_omega.abs() < 1e-6, lerped, slerped)
    interpolated = interpolated.view(NUM_PAIRS, NUM_INTERPOLATIONS, *latents.shape[1:])
    return interpolated.view(NUM_PAIRS * NUM_INTERPOLATIONS, *latents.shape[1:])


def main():
    device = get_device()
    print(f"device={device}")
    torch.manual_seed(42)

    betas = build_variance_schedule(device, T)
    _, alphas_bar = build_alpha(betas)
    steps = build_ddim_schedule(T, NUM_STEPS)

    model = NaiveUnet(in_channels=1, out_channels=1, n_feat=64).to(device)
    ckpt_path = os.path.join(_dirname, "../imgDDPM/outputs/ckpt.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cache_path = os.path.join(_dirname, "../imgDDPM/.cache/mnist_data_train.pt")
    loader = load_mnist(NUM_PAIRS * 2, cache_path=cache_path, shuffle=True)
    x_0 = next(iter(loader))[0].to(device)

    with torch.no_grad():
        latents = inversion(model, x_0, steps, alphas_bar)
        interpolated_latents = interpolate_pairs(latents)
        images = denoise(model, interpolated_latents, steps, alphas_bar)

    save_image_grid(
        images.clamp(-1.0, 1.0),
        os.path.join(_dirname, f"outputs/ddim_interpolate_{NUM_STEPS}.png"),
        image_size=IMG_SIZE,
        nrow=NUM_INTERPOLATIONS,
    )


if __name__ == "__main__":
    main()
