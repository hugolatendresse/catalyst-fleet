"""
Model Type: only layernorm
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

# Create a dummy model with LayerNorm
class PyTorchLayerNorm(nn.Module):
    def __init__(self):
        super(PyTorchLayerNorm, self).__init__()
        # For an input of shape [batch, 2, 2, 2],
        # we normalize over the last three dimensions.
        self.ln = nn.LayerNorm([2, 2, 2])

    def forward(self, x):
        return self.ln(x)

torch_model = PyTorchLayerNorm().eval()

raw_data = np.random.rand(1, 2, 2, 2).astype("float32")

test_export_and_cuda(raw_data, torch_model)
