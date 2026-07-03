import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


T = 500
T_EMB_DIM = 16

MUS = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
SIGS = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
PI = torch.tensor([0.4, 0.6])


def sample_data(n):
    component_ids = torch.multinomial(PI, n, replacement=True)
    return MUS[component_ids] + torch.sqrt(SIGS[component_ids]) * torch.randn(n, 2)


def build_variance_schedule():
    betas = torch.zeros(T + 1, 1)
    betas[1:] = torch.linspace(1e-3, 0.02, T).view(-1, 1)
    return betas


def build_alpha(betas):
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas, alphas_bar


betas = build_variance_schedule()
alphas, alphas_bar = build_alpha(betas)


def q_sample(x0, t, noise):
    return torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1.0 - alphas_bar[t]) * noise


class Toyfusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
        )
        self.timestep_layer = nn.Linear(T_EMB_DIM, 32)
        self.output_layers = nn.Sequential(
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )

    def forward(self, x_t, t):
        t_emb = self._get_timestep_embedding(t)
        h = self.data_layers(x_t) + self.timestep_layer(t_emb)
        return self.output_layers(h)

    def _get_timestep_embedding(self, t):
        t = t.float().reshape(-1, 1)
        half_dim = T_EMB_DIM // 2
        frequencies = torch.exp(
            -np.log(10000) * torch.arange(half_dim, dtype=t.dtype) / half_dim
        )
        angles = t * frequencies.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def train(model, steps, batch_size, lr):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x0 = sample_data(batch_size)
        noise = torch.randn_like(x0)
        t = torch.randint(1, T + 1, (batch_size,))
        x_t = q_sample(x0, t, noise)

        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == steps - 1:
            print(f"step={step}, loss={loss.item():.4f}")


def p_sample(model, x_t, t):
    noise_pred = model(x_t, t)
    return (x_t - betas[t] / torch.sqrt(1.0 - alphas_bar[t]) * noise_pred) / torch.sqrt(
        alphas[t]
    )


def mixture_density(xx, yy):
    points = np.stack([xx, yy], axis=-1)
    density = np.zeros(xx.shape)
    for weight, mean, var in zip(PI.numpy(), MUS.numpy(), SIGS.numpy()):
        diff = points - mean
        norm = 1.0 / (2.0 * np.pi * np.sqrt(np.prod(var)))
        exponent = -0.5 * np.sum(diff ** 2 / var, axis=-1)
        density += weight * norm * np.exp(exponent)
    return density


def render_frame(points, timestep):
    points = points.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    x = np.linspace(-4.5, 4.5, 200)
    y = np.linspace(-4.5, 4.5, 200)
    xx, yy = np.meshgrid(x, y)
    density = mixture_density(xx, yy)
    low_levels = np.geomspace(density.max() * 1e-5, density.max() * 0.03, 6)
    high_levels = np.linspace(density.max() * 0.05, density.max() * 0.95, 8)
    levels = np.concatenate([low_levels, high_levels])
    ax.contour(xx, yy, density, levels=levels, cmap="plasma", linewidths=0.8)
    ax.scatter(points[:, 0], points[:, 1], s=1.0, c="black", alpha=0.6)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"t = {timestep}")
    fig.tight_layout()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


@torch.no_grad()
def sample_gif(model, n, save_every, gif_path):
    model.eval()
    frames = []
    x_t = torch.randn(n, 2)

    for i in range(T, 0, -1):
        t = torch.full((n,), i, dtype=torch.long)
        mu = p_sample(model, x_t, t)
        x_t = mu + torch.sqrt(betas[t]) * torch.randn_like(x_t) if i != 1 else mu

        if i % save_every == 0 or i == 1:
            frames.append(render_frame(x_t, i))

    imageio.mimsave(gif_path, frames, duration=0.08)
    print(f"saved {gif_path}")
    return x_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--gif-path", type=str, default="sampling.gif")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    model = Toyfusion()
    train(model, args.steps, args.batch_size, args.lr)
    sample_gif(model, args.num_samples, args.save_every, args.gif_path)


if __name__ == "__main__":
    main()
