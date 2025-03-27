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


# Create a dummy model
class IndexModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x[0,2]
        return z
        
torch_model = IndexModel().eval()

raw_data = np.random.rand(10,10,10).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)