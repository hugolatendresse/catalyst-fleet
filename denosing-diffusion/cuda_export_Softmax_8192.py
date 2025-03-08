"""
Model Type: softmax op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
from torch import nn
import numpy as np

from torch.nn import Softmax


import torch
import torch.nn as nn

# class SafeSoftmax(nn.Module):
#     """
#     A safe implementation of softmax that handles large tensors with non-last dimensions.
#     This is a workaround for a bug in TVM where softmax on non-last dimensions with large
#     tensors causes CUDA shared memory allocation issues.
#     """
#     def __init__(self, dim=None):
#         super().__init__()
#         self.dim = dim
#         self.softmax = nn.Softmax(dim=dim)
    
#     def forward(self, x):
#         if self.dim is None or self.dim == -1 or self.dim == len(x.shape) - 1:
#             # For last dimension softmax, use the standard implementation
#             return self.softmax(x)
        
#         # For non-last dimensions with large tensors, use the transpose workaround
#         input_shape = x.shape
        
#         # Check if tensor is large (more than 4096 in any dimension)
#         is_large = any(s > 4096 for s in input_shape)
        
#         if is_large:
#             # Get the dimension ordering for transpose
#             # Move the softmax dimension to the end
#             dims = list(range(len(input_shape)))
#             dims.append(dims.pop(self.dim))
            
#             # Transpose to move softmax dim to the end
#             x_transposed = x.permute(dims)
            
#             # Apply softmax on the last dimension (which is our target dimension)
#             y_transposed = torch.nn.functional.softmax(x_transposed, dim=-1)
            
#             # Transpose back to original shape
#             # Calculate the inverse permutation
#             inv_dims = [-1] * len(dims)
#             for i, d in enumerate(dims):
#                 inv_dims[d] = i
            
#             y = y_transposed.permute(inv_dims)
#             return y
#         else:
#             # For smaller tensors, use the standard implementation
#             return self.softmax(x)


torch_model = Softmax(dim=2).eval()



raw_data = np.random.rand(1, 4, 32, 8192).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model) 