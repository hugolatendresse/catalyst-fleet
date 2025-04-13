"""
Model Type: 
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

class BroadcastTensorsModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.tensor([1, 2, 3])         # Shape: [3]
        return  y.expand_as(x)


torch_model = BroadcastTensorsModel().eval()

raw_data = np.random.rand(2,3).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)