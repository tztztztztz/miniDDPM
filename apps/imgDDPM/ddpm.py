import numpy as np
import torch


# Initialize the ddpm hyperparameters
def build_variance_schedule(device, timesteps):
    betas = torch.zeros(timesteps + 1, 1, 1, 1, device=device)
    betas[1:] = torch.linspace(1e-4, 0.02, timesteps, device=device).view(-1, 1, 1, 1)
    return betas

def build_alpha(betas):
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas, alphas_bar


#  Compute the forward process of the diffusion model
def q_sample(x0, t, noise, alphas_bar):
    return torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1.0 - alphas_bar[t]) * noise


#  Compute the reverse process of the diffusion model
def p_mean(model, x_t, t, betas, alphas, alphas_bar):
    noise_pred = model(x_t, t)
    return (x_t - betas[t] / torch.sqrt(1.0 - alphas_bar[t]) * noise_pred) / torch.sqrt(
        alphas[t]
    )

def p_sample(model, x_t, t, betas, alphas, alphas_bar):
    mu = p_mean(model, x_t, t, betas, alphas, alphas_bar)
    return mu + torch.sqrt(betas[t]) * torch.randn_like(x_t) if t[0].item() != 1 else mu


# Iterate over the timesteps to sample the images
@torch.no_grad()
def sample_images(
    model,
    n,
    image_size,
    device,
    betas,
    alphas,
    alphas_bar,
    timesteps,
):
    model.eval()
    frames = []
    save_timesteps = set(np.linspace(timesteps, 10, 20).round().astype(int).tolist())
    save_timesteps.update(range(9, -1, -1))

    x_t = torch.randn(n, 1, image_size, image_size, device=device)
    if timesteps in save_timesteps:
        frames.append((timesteps, x_t.clamp(-1.0, 1.0).detach().cpu()))

    for i in range(timesteps, 0, -1):
        t = torch.full((n,), i, dtype=torch.long, device=device)
        x_t = p_sample(model, x_t, t, betas, alphas, alphas_bar)

        if i - 1 in save_timesteps:
            frames.append((i - 1, x_t.clamp(-1.0, 1.0).detach().cpu()))

    return x_t.clamp(-1.0, 1.0), frames
