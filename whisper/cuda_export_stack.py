"""
Model Type: stack.default
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: FAIL
Correctness Test: FAIL
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
class StackModel(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        val1 = x[1,4]
        val2 = x[3,2]
        val3 = x[5,6]
        z = torch.stack([val1, val2, val3])
        return z
        

torch_module = StackModel().eval()

raw_data = np.random.rand(10,10,10).astype("float32")

test_export_and_cuda(raw_data, torch_module)