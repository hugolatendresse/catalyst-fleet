"""
Model Type: sum.default
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


# Create a dummy model
class SumModel(nn.Module):
    def forward(self, x):
        new_vec = x[1,4]
        return new_vec.sum()
        
torch_model = SumModel().eval()

raw_data = np.random.rand(10,10,10).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)