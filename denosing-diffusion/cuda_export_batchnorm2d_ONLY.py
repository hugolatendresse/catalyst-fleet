"""
Model Type: only batchnorm1D
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: FAIL (can't handle batchnorm.default)
Correctness Test: FAIL
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

import torch.nn.functional as F
import numpy as np
from hlutils.test_export_and_cuda import test_export_and_cuda
from hlutils.set_seed_all import set_seed_all

set_seed_all()

# Create a dummy model
class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)  # BatchNorm after conv1

    def forward(self, x):
        return self.bn1(x)

torch_model = PyTorchCNN().eval()

raw_data = np.random.rand(1, 3, 128, 128).astype("float32")

test_export_and_cuda(raw_data, torch_model)