"""
Model Type: 
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

class BroadcastShapeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = torch.broadcast_shapes(torch.Size([3]), torch.Size([1,3]))
        return  x.reshape(shape)


"""
>>> import torch
>>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))                   pHelper<tvm::runtime::PackedFuncValueConverter<tvm::runtime::Array<tvm
>>> x=torch.tensor([1,2,3])
>>> y=torch.tensor([[1,2,3]])
>>> shape = torch.broadcast_shapes(x.shape, y.shape)
>>> shape
torch.Size([1, 3])                 ime::Array<tvm::RelaxExpr, void> >::From(tvm::runtime::TVMArgValue con
>>> x.reshape(shape)
tensor([[1, 2, 3]])
>>> 
"""

torch_model = BroadcastShapeModel().eval()

raw_data = np.random.rand(3).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)