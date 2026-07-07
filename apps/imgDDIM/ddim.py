import torch

# We assume we set sigma = 0, so the process is deterministic.
def build_ddim_schedule(T, num_steps):
    # subsample the timesteps (linearly)
    steps = torch.linspace(1, T, steps=num_steps).round().long().tolist()
    return [0] + steps
    
def p_mean(model, x_t, t, prev_t, alphas_bar):
    noise_pred = model(x_t, t)
    x_0 = (x_t - torch.sqrt(1.0 - alphas_bar[t]) * noise_pred) / torch.sqrt(alphas_bar[t])
    if prev_t[0].item() == 0:
        return x_0
    return torch.sqrt(alphas_bar[prev_t]) * x_0 + torch.sqrt(1.0 - alphas_bar[prev_t]) * noise_pred

def p_sample(model, x_t, t, prev_t, alphas_bar):
    mu = p_mean(model, x_t, t, prev_t, alphas_bar)
    return mu # we assume sigma = 0, so the process is deterministic


def inversion(model, x_0, steps, alphas_bar):
    """
    
    steps: the timesteps to invert the image. 
        Like: [0, 1, 2, ..., T-1, T] 
    
    Invert the image x_0 to the latent noise x_T, using DDIM inversion.
    Variance sigma is set to 0.
    """
    x_prev = x_0
    batch_size = x_0.shape[0]
    device = x_0.device

    for i in range(len(steps) - 1):
        prev_t = torch.full((batch_size,), steps[i], dtype=torch.long, device=device)
        t = torch.full((batch_size,), steps[i + 1], dtype=torch.long, device=device)

        noise_pred = model(x_prev, t) # this is an approximation.
        alpha_bar_t = alphas_bar[t]
        alpha_bar_prev = alphas_bar[prev_t]

        """
        Here, we use a **single formula** for t != 1 and t = 1.
        Because we set alpha_bar_t[0] = 1
        """

        x_t = (
            torch.sqrt(alpha_bar_t / alpha_bar_prev) * x_prev
            + torch.sqrt(alpha_bar_t)
            * (
                torch.sqrt(1.0 / alpha_bar_t - 1.0)
                - torch.sqrt(1.0 / alpha_bar_prev - 1.0)
            )
            * noise_pred
        )

        x_prev = x_t

    return x_t
