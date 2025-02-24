"""
Model Type: only groupnorm
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python')
import numpy as np
from torch import nn
from hlutils.test_export_and_cuda import test_export_and_cuda
from hlutils.set_seed_all import set_seed_all

set_seed_all()

class PyTorchGroupNorm(nn.Module):
    def __init__(self):
        super(PyTorchGroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=2, num_channels=6)

    def forward(self, x):
        return self.gn(x)

torch_model = PyTorchGroupNorm().eval()

raw_data = np.random.rand(10, 6, 28, 28).astype("float32")

test_export_and_cuda(raw_data, torch_model)
