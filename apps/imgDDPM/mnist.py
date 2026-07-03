import os

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image


IMG_SIZE = 28
DEFAULT_CACHE_PATH = os.path.join(".cache", "mnist_data_train.pt")


def build_mnist_tensor_cache(cache_path: str = DEFAULT_CACHE_PATH) -> torch.Tensor:
    dataset = load_dataset("ylecun/mnist", split="train")
    images = []

    for item in dataset:
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("L")
        image = np.array(image, dtype=np.float32) / 127.5 - 1.0
        images.append(image)

    x = torch.from_numpy(np.stack(images))[:, None, :, :]
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(x, cache_path)
    print(f"saved MNIST tensor cache: {cache_path}, shape={tuple(x.shape)}")
    return x


def load_mnist(
    batch_size: int,
    cache_path: str = DEFAULT_CACHE_PATH,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    if os.path.exists(cache_path):
        x = torch.load(cache_path, map_location="cpu")
        print(f"loaded MNIST tensor cache: {cache_path}, shape={tuple(x.shape)}")
    else:
        x = build_mnist_tensor_cache(cache_path)

    dataset = torch.utils.data.TensorDataset(x)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
