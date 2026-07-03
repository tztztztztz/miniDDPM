# DDPM Bug Notes

This file records the main bugs and implementation pitfalls found while building the tiny 2D DDPM.

## 1. Forward Diffusion Missed Square Roots

Wrong version:

```python
xt = alphas_bar[t] * x0 + (1 - alphas_bar[t]) * noise
```

Correct version:

```python
xt = torch.sqrt(alphas_bar[t]) * x0 + torch.sqrt(1 - alphas_bar[t]) * noise
```

Reason:

The closed-form DDPM forward process is:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

`alpha_bar_t` and `1 - alpha_bar_t` are variances. The coefficients applied to samples must be standard deviations, so they need square roots.

## 2. Beta Schedule Was Too Aggressive

Wrong version:

```python
betas[1:] = np.linspace(1e-3, 1 - 1e-3, T)
```

Better version:

```python
betas[1:] = np.linspace(1e-3, 0.02, T)
```

Reason:

Setting the largest beta close to `1` destroys almost all signal in a single step. Then `alpha_bar_t` collapses too quickly, and most timesteps are basically pure noise.

For a toy DDPM, beta should stay small so the forward process gradually corrupts the data.

## 3. Reverse Sampling Reinitialized Noise Inside the Loop

Wrong version:

```python
for i in range(T, 0, -1):
    xt = torch.randn((N, 2))
    ...
```

Correct version:

```python
xt = torch.randn((N, 2))
for i in range(T, 0, -1):
    ...
```

Reason:

The reverse process is a Markov chain from `x_T` to `x_0`. Reinitializing `xt` inside every step breaks the chain completely.

## 4. Reverse Sampling Used Uniform Noise

Wrong version:

```python
torch.rand_like(xt)
```

Correct version:

```python
torch.randn_like(xt)
```

Reason:

The reverse transition is Gaussian. The injected noise should be sampled from a standard normal distribution, not a uniform distribution.

## 5. Reverse Noise Scale Used Variance Instead of Standard Deviation

Wrong version:

```python
xt = mu + betas[t] * torch.randn_like(xt)
```

Acceptable toy version:

```python
xt = mu + torch.sqrt(betas[t]) * torch.randn_like(xt)
```

Reason:

If the reverse variance is `beta_t`, then the noise multiplier should be the standard deviation `sqrt(beta_t)`.

Note:

The more exact DDPM sampler can use posterior variance instead of `beta_t`, but `sqrt(beta_t)` is acceptable for this tiny toy implementation.

## 6. Sampling Built Computation Graphs

Wrong pattern:

```python
xt = torch.randn((N, 2))
for i in range(T, 0, -1):
    noise_pred = model(xt, t)
    ...
```

Correct pattern:

```python
model.eval()

@torch.no_grad()
def sample_ddpm(N=1000):
    xt = torch.randn((N, 2))
    for i in range(T, 0, -1):
        ...
    return xt
```

Reason:

Sampling is inference. Without `torch.no_grad()`, PyTorch stores unnecessary computation graphs for all reverse steps, wasting memory and time.

## 7. GIF Frame Order Was Reversed

Wrong version:

```python
frame_paths = frame_paths[::-1]
```

Correct version:

```python
images = [imageio.imread(path) for path in frame_paths]
imageio.mimsave(gif_path, images, duration=0.08)
```

Reason:

The loop already saves frames in sampling order:

```text
t = T, T - 1, ..., 1
```

That is the desired animation direction: noise to data. Reversing `frame_paths` makes the GIF play backward.

## 8. GIF Axes Were Not Fixed

Problem:

Each frame could autoscale its axes differently, making the GIF visually jump.

Fix:

```python
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
```

Reason:

All frames should use the same coordinate system so the denoising trajectory is visually comparable.
