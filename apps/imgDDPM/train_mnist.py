import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from ddpm import (
    build_alpha,
    build_variance_schedule,
    q_sample,
    sample_images,
)
from mnist import load_mnist
from unet import NaiveUnet
from utils import save_gif, save_image_grid

IMG_SIZE = 28
T = 1000


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"loaded {path}")


def train(model, loader, steps, lr, device, alphas_bar, timesteps):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    data_iter = iter(loader)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x0 = batch[0].to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(1, timesteps + 1, (x0.shape[0],), device=device)
        x_t = q_sample(x0, t, noise, alphas_bar)

        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            print(f"step={step}, loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-path", type=str, default="outputs/final_samples.png")
    parser.add_argument("--gif-path", type=str, default="outputs/sampling.gif")
    parser.add_argument("--ckpt-path", type=str, default="outputs/ckpt.pt")
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only and args.load_from is None:
        parser.error("--eval-only needs --load-from")

    torch.manual_seed(0)
    np.random.seed(0)

    device = get_device()
    print(f"device={device}")

    betas = build_variance_schedule(device, T)
    alphas, alphas_bar = build_alpha(betas)

    model = NaiveUnet(in_channels=1, out_channels=1, n_feat=64).to(device)

    if args.load_from is not None:
        load_checkpoint(model, args.load_from, device)

    if not args.eval_only:
        loader = load_mnist(args.batch_size)
        train(model, loader, args.steps, args.lr, device, alphas_bar, T)

        os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.ckpt_path)
        print(f"saved {args.ckpt_path}")

    images, frames = sample_images(
        model,
        args.num_samples,
        IMG_SIZE,
        device,
        betas,
        alphas,
        alphas_bar,
        T,
    )
    save_image_grid(images, args.sample_path)
    save_gif(frames, args.gif_path)


if __name__ == "__main__":
    main()
