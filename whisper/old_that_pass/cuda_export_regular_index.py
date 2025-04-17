"""
Model Type: index.Tensor
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
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

class IndexModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[[0]]
        # return x[1,2] # select.int; select.int; passes correctness
        # return x[1,2:4] # select.int; slice.Tensor; passes correctness
        
torch_model = IndexModel().eval()

raw_data = np.random.rand(5,5,5).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)