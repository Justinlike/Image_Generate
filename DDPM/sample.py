from diffusers import UNet2DModel
from diffusers.utils import make_image_grid, numpy_to_pil
import torch.nn.functional as F
import os
from data import config
from model import DDPM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ddpm = DDPM()
model = UNet2DModel.from_pretrained("ddpm-animefaces-64").to(device)
model.eval()

with torch.no_grad():
    images = ddpm.sample(model, config.eval_batch_size, 3, config.image_size)

print(images.shape)
pil_images = numpy_to_pil(images)  # 先把 numpy 转成 PIL 列表
image_grid = make_image_grid(pil_images, rows=4, cols=4)
# image = numpy_to_pil(image_grid)
os.makedirs(os.path.join(config.output_dir, "samples_out"), exist_ok=True)
image_grid.save(os.path.join(config.output_dir, "samples_out", "sample.png"))