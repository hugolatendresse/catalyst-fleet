"""
Model Type: chunk op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ?
Correctness Test: ?
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
import numpy as np
import tvm
import math
import torch
from torch import nn

batch = 3
channels = 139
height = 7
width = 11

chunks = 42
dim = 1

raw_data = np.random.rand(batch, channels, height, width).astype("float32")

class ChunkModel(nn.Module):
    def __init__(self, chunks, dim):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x):
        return x.chunk(self.chunks, dim=self.dim)


class SplitModel(nn.Module):
    def __init__(self, split_size, dim):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

torch_model_chunk = ChunkModel(chunks=chunks, dim=dim).eval()
split_size = math.ceil(raw_data.shape[dim] / chunks)
torch_model_split = SplitModel(split_size=split_size, dim=dim).eval()

torch_data = torch.from_numpy(raw_data)

torch_output_chunk = torch_model_chunk(torch_data)
torch_output_split = torch_model_split(torch_data)
print("chunk torch_output has length", len(torch_output_chunk))
assert len(torch_output_chunk) == len(torch_output_split), f"different lengths!!. chunk: {len(torch_output_chunk)}, split: {len(torch_output_split)}"
# desired_chunk_0 = torch_output_chunk[0].detach().numpy()
# desired_chunk_1 = torch_output_chunk[1].detach().numpy()
# desired_split_0 = torch_output_split[0].detach().numpy()
# desired_split_1 = torch_output_split[1].detach().numpy()

for i in range(len(torch_output_chunk)):
    assert torch_output_chunk[i].shape == torch_output_split[i].shape, "different shapes!!"
    np.testing.assert_allclose(actual=torch_output_chunk[i].detach().numpy(), desired=torch_output_split[i].detach().numpy(), rtol=1e-5, atol=1e-5)
    print("matches!")


