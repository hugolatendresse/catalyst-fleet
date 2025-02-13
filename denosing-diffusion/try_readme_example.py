"""
Diffusion model from # https://github.com/lucidrains/denoising-diffusion-pytorch
This example is taken from the README file


"""


import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 10    # number of steps
)
from torch import nn
assert isinstance(diffusion, nn.Module), "Our methodology expects to ingest an nn.Module"

training_images = torch.rand(1, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)