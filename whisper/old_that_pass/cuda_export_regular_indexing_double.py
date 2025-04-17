"""
Model Type: index.Tensor
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
"""
import numpy as np
import torch
from torch import nn
import numpy as np

class RegularIndexingSingle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[3,1] # Exports as select.int twice
        
torch_model = RegularIndexingSingle().eval()

raw_data = np.random.rand(5,5,5).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)