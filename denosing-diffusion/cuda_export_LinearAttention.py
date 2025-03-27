"""
Model Type: CNN
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: SUCCESS
Correctness Test: SUCCESS
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

def my_chunk(t, chunks, dim = 0):
    split_size_or_sections = math.ceil(t.size(dim) // chunks)
    out = torch.split(t, split_size_or_sections, dim)
    return out


# Create a dummy model

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

    def forward(self, x):
        
        print("x.shape", x.shape)


        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = my_chunk(qkv, 3, dim = 1) # TODO restore 
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        return q


# shape of stuf in init of LinearAttentoin 64 4 32 4
# shape of stuf in init of LinearAttentoin 64 4 32 4
# shape of stuf in init of LinearAttentoin 128 4 32 4
# A100 GPU detected, using flash attention if input tensor is on cuda
# shape of stuf in init of LinearAttentoin 256 4 32 4
# shape of stuf in init of LinearAttentoin 128 4 32 4
# shape of stuf in init of LinearAttentoin 64 4 32 4
# x.shape in forward of LinearAttention torch.Size([1, 64, 128, 128]


torch_model = LinearAttention(dim=64, heads=4, dim_head=32, num_mem_kv=4).eval()

raw_data = np.random.rand(1, 64, 128, 128).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model) 