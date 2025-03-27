"""
Model Type: index.Tensor
Model Definition: PyTorch
Model Export: fx
Target: CUDA
Compile and Run Test: FAILS, TypeError: 'Node' object is not iterable in translator
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
        self.position_ids = torch.tensor([0])

    def forward(self, x):
        return x[self.position_ids]
        
torch_model = IndexModel().eval()

raw_data = np.random.rand(3,3).astype("float32")

from hlutils import test_fx_and_cuda
input_info = [((3,3), "float32")]
test_fx_and_cuda(raw_data, torch_model, input_info=input_info)