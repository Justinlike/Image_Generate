from diffusers import UNet2DModel
from diffusers.utils import make_image_grid, numpy_to_pil
import torch.nn.functional as F
import os
from data import dataloader, config
from model import DDPM
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

ddpm = DDPM()
model = UNet2DModel.from_pretrained(config.model_path).to(device)
model.eval()

with torch.no_grad():
    images = ddpm.sample(model, config.eval_batch_size, 3, config.image_size)

image_grid = make_image_grid(images, nrow=int(config.eval_batch_size ** 0.5))
image = numpy_to_pil(image_grid)
os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
image_grid.save(os.path.join(config.output_dir, "samples", "sample.png"))