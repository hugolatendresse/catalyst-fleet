"""
Model Type: index.Tensor
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ??
Correctness Test: ??
"""
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import torch.nn.functional as F
import numpy as np

"""
Trying to achieve:
self.weight[position_ids]

With:
self.weight.shape = torch.Size([448, 384])
position_ids.shape  = torch.Size([1, 1])
position_ids = tensor([[0]])

In general, torch does this:
>>> import torch
>>> x = torch.Tensor([[10,11],[12,13]])
>>> x[torch.tensor([[0]])]
tensor([[[10., 11.]]])
>>> x[[[0]]]
tensor([[10., 11.]])

"""

# Create a dummy model
class IndexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = torch.tensor([[0]])

    def forward(self, x):
        return x[self.position_ids]
        
torch_model = IndexModel().eval()

raw_data = np.random.rand(2,3).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)