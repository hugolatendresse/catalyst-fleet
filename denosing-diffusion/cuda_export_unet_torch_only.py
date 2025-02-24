"""
Model Type: simple unet
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ??
Correctness Test: ??
"""
import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

import torch
from denoising_diffusion_pytorch import Unet

# TODO HL commented out this assertion
# assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

torch_model = model.eval()

raw_data = np.random.rand(1,3,128, 128).astype("float32")

torch_data = torch.from_numpy(raw_data)

batch_size = 1
num_timesteps = 10
time_data = torch.randint(0, num_timesteps, (batch_size,), dtype=torch.long)

# Give an example argument to torch.export

example_args = (torch_data, time_data)


pytorch_out = torch_model(torch_data, time_data).detach().numpy() 
print(pytorch_out)


