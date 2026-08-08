from diffusers import UNet2DModel
from data import config
import torch
from tqdm import tqdm

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ), 
)

class DDPM:
    def __init__(self,
                 num_train_steps:int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 ) -> None:
        self.betas = torch.linspace(beta_start, beta_end, num_train_steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_steps - 1, -1, -1)
        self.num_train_timesteps = num_train_steps
    
    def add_noise(
            self,
            orignial_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor
        ):
        alphas_cumprod = self.alphas_cumprod.to(device=orignial_samples.device, dtype=orignial_samples.dtype)
        noise = noise.to(orignial_samples.device)
        timesteps = timesteps.to(orignial_samples.device)

        # sqrt(bar (alpha_t))
        sqrt_alphas_prod = torch.sqrt(alphas_cumprod[timesteps])[:, None, None, None]
        while len(sqrt_alphas_prod.shape) < len(orignial_samples.shape):
            sqrt_alphas_prod = sqrt_alphas_prod.unsqueeze(-1)
        
        # sqrt(1 - bar (alpha_t))
        sqrt_one_minus_alpha_prod = torch.sqrt((1.0 - alphas_cumprod[timesteps])[:, None, None, None])
        while len(sqrt_one_minus_alpha_prod.shape) < len(orignial_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        return sqrt_alphas_prod * orignial_samples + sqrt_one_minus_alpha_prod

    @torch.no_grad()
    def sample(
            self,
            unet: UNet2DModel,
            batch_size: int,
            in_channels: int,
            sample_size: int,
        ):
        betas = self.betas.to(unet.device)
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)

        for timestep in tqdm(timesteps, desc="Sampling"):
            pred_noise: torch.Tensor = unet(images, timestep).sample

            # mean of q(x_{t-1}|x_t)
            alpha_t = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            one_minus_alpha_t = 1.0 - alpha_t
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            mean = (images - one_minus_alpha_t / sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_t
            
            # var of q(x_{t-1}|x_t)
            if timestep > 0:
                beta_t = betas[timestep]
                one_minus_alpha_cumprod_t_minus_one = 1.0 - alphas_cumprod[timestep - 1]
                one_divided_by_sigma_square = alpha_t / beta_t + 1.0 / one_minus_alpha_cumprod_t_minus_one
                variance = torch.sqrt(1.0 / one_divided_by_sigma_square)
            else:
                variance = torch.zeros_like(timestep)
            
            epsilon = torch.randn_like(images)
            images = mean + variance * epsilon
        
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images
    

    