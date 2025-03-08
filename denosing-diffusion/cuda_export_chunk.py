"""
Model Type: chunk op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ?
Correctness Test: ?
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
from torch import nn
import numpy as np

from torch.nn import Softmax

raw_data = np.random.rand(3, 4, 7, 11).astype("float32")

class ChunkModel(nn.Module):
    def __init__(self, chunks, dim):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x):
        return x.chunk(self.chunks, dim=self.dim)

torch_model = ChunkModel().eval()

from hlutils.test_export_and_cuda import test_export_and_cuda
test_export_and_cuda(raw_data, torch_model) 