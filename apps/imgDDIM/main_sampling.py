import sys
import os

import torch

_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_dirname, "../imgDDPM"))


T = 1000
NUM_STEPS = 100
IMG_SIZE = 28

from ddim import build_ddim_schedule, p_sample
from ddpm import build_variance_schedule, build_alpha
from unet import NaiveUnet
from utils import save_image_grid


device = "cuda"
batch_size = 64
if not torch.cuda.is_available():
    device = "cpu"

betas = build_variance_schedule(device, T)
alphas, alphas_bar = build_alpha(betas)
steps = build_ddim_schedule(T, NUM_STEPS)

model = NaiveUnet(in_channels=1, out_channels=1, n_feat=64).to(device)
model.load_state_dict(
    torch.load(os.path.join(_dirname, "../imgDDPM/outputs/ckpt.pt"), map_location=device)
)
model.eval()

step_inv = steps[::-1]

x_t = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE).to(device)
with torch.no_grad():
    for i in range(len(step_inv)-1):
        t = torch.full((batch_size,), step_inv[i], dtype=torch.long, device=device)
        prev_t = torch.full((batch_size,), step_inv[i+1], dtype=torch.long, device=device)
        x_t = p_sample(model, x_t, t, prev_t, alphas_bar)

save_image_grid(x_t.clamp(-1.0, 1.0).detach().cpu(), os.path.join(_dirname, "outputs/ddim_sampling_mnist.png"))
