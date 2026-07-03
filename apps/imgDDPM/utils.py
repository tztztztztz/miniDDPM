import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from mnist import IMG_SIZE


def render_image_grid(images, image_size=IMG_SIZE, nrow=8):
    images = (images.detach().cpu() + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    n = images.shape[0]
    ncol = int(np.ceil(n / nrow))
    grid = torch.ones(1, ncol * image_size, nrow * image_size)

    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[
            :,
            row * image_size : (row + 1) * image_size,
            col * image_size : (col + 1) * image_size,
        ] = image

    array = (grid.squeeze(0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_image_grid(images, path, image_size=IMG_SIZE, nrow=8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    render_image_grid(images, image_size=image_size, nrow=nrow).save(path)
    print(f"saved {path}")


def save_gif(frames, path, image_size=IMG_SIZE, nrow=8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = [
        np.asarray(render_image_grid(images, image_size=image_size, nrow=nrow))
        for _, images in frames
    ]
    imageio.mimsave(path, images, duration=0.1, loop=0)
    print(f"saved {path}")
