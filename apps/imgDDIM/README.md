# DDIM

## Result

### Sampling

![DDIM sampling](assets/sampling.png)

### DDIM Inversion Reconstruction

Top row: original MNIST images. Bottom row: reconstructed images after DDIM inversion and denoising.

![DDIM inversion reconstruction](assets/reconstruction.png)

### Latent Interpolation

Each row interpolates between two inverted MNIST latents, then denoises the interpolated latents.


linear interpolation:

![DDIM latent interpolation](assets/interpolate_linear.png)

slerp interpolation:

![DDIM latent interpolation](assets/interpolate_slerp.png)

## Run

    python main_sampling.py

## Reference

- [From DDPM to DDIM](https://deepschool.ai/blog/2024-02-11-DDPM-to-DDIM.html)

- https://zhuanlan.zhihu.com/p/565698027

- https://zhuanlan.zhihu.com/p/627616358
