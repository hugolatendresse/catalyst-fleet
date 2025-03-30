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


import torch
import operator
from functools import reduce

def _prod(shape):
    """Compute the product of all dimensions in 'shape'."""
    out = 1
    for s in shape:
        out *= s
    return out

def _broadcast_shapes(shapes):
    """
    Minimal re-implementation of torch.broadcast_shapes for older PyTorch.
    For new versions, you can do: return torch.broadcast_shapes(*shapes)
    """
    max_ndim = max(len(s) for s in shapes)
    rev_shapes = [s[::-1] for s in shapes]
    out = []
    for i in range(max_ndim):
        dim_size = 1
        for rsh in rev_shapes:
            if i < len(rsh):
                s_ = rsh[i]
                if s_ != 1 and dim_size != 1 and s_ != dim_size:
                    raise ValueError(f"Incompatible shapes for broadcast: {shapes}")
                dim_size = max(dim_size, s_)
        out.append(dim_size)
    out.reverse()
    return tuple(out)

def _is_multiple_indices(index_arg):
    """
    Decide if 'index_arg' is a 'multiple parallel indices' scenario
    vs. a single advanced index. For the examples:

      - If top-level is a list/tuple and len(index_arg) > 1,
        interpret it as multiple indices (like [ [0,2], [1,3] ]).
      - Otherwise, it's a single advanced index (like [[[0,2],[1,3]]]).
    """
    if isinstance(index_arg, (list, tuple)):
        if len(index_arg) > 1:
            return True
    return False

def transform(t, index_arg):
    """
    Replicate t[index_arg] using:
      - basic (scalar) indexing on 't'
      - slicing
      - torch.index_select
      - cat/stack
      - broadcasting
    … but NOT using advanced indexing on 't'.

    We do use advanced indexing on a small dummy 1D array to discover
    the correct 'leading shape' for single-index scenarios, which is
    the official Python-list fancy indexing logic PyTorch uses.
    """

    # -----------------------------------------------------------
    # CASE B: multiple advanced indices (top-level length > 1)
    # -----------------------------------------------------------
    if _is_multiple_indices(index_arg):
        # Example: t[[0,2], [1,3]] => parallel indices for dimension0 and dimension1
        idx_list = []
        for sub_i in index_arg:
            idx_list.append(torch.tensor(sub_i, dtype=torch.long))

        # 1) Broadcast them all to a common shape B
        shapes = [x.shape for x in idx_list]
        B = _broadcast_shapes(shapes)
        # 2) Expand each index to that shape
        for i in range(len(idx_list)):
            idx_list[i] = idx_list[i].expand(B)

        k = len(idx_list)                # number of advanced dimensions
        leftover_dims = t.shape[k:]      # leftover dims after those k

        M = _prod(B)  # total number of elements in the broadcast
        # 3) Flatten each index to shape [M]
        for i in range(k):
            idx_list[i] = idx_list[i].reshape(M)

        # 4) Enumerate each of the M broadcasted coordinates, do scalar indexing on 't'
        slices = []
        for n in range(M):
            out_slice = t
            for i in range(k):
                scalar_idx = idx_list[i][n].item()
                out_slice = out_slice[scalar_idx]
            # shape leftover_dims
            slices.append(out_slice.unsqueeze(0))  # shape [1, leftover_dims]

        # 5) Concatenate -> shape [M, leftover_dims]
        stacked = torch.cat(slices, dim=0)
        # 6) Reshape to [B, leftover_dims]
        final_shape = list(B) + list(leftover_dims)
        result = stacked.view(*final_shape)
        return result

    # -----------------------------------------------------------
    # CASE A: single advanced index
    # -----------------------------------------------------------
    # We'll use the "dummy approach" to discover how PyTorch shapes
    # the advanced index on dimension=0.
    #
    #   1) Make dummy = torch.arange(t.shape[0])
    #   2) Let dummy_selected = dummy[index_arg]  (Python advanced indexing)
    #   3) Flatten -> real indices
    #   4) index_select(t, dim=0, those_indices)
    #   5) reshape => [dummy_selected.shape, leftover_dims]

    dummy = torch.arange(t.shape[0])  # shape [t.shape[0]]
    # Use the same Python-list advanced indexing on dummy
    dummy_selected = dummy[index_arg]  
    # Now dummy_selected.shape is exactly how PyTorch interprets
    # the advanced index shape for dimension 0.

    adv_shape = dummy_selected.shape  # e.g. [1] or [2,2], etc.
    # Flatten to get real indices
    flattened = dummy_selected.reshape(-1)  # shape [M]
    # Gather from t along dim=0
    picked = torch.index_select(t, dim=0, index=flattened)
    # leftover dims
    leftover_dims = t.shape[1:]
    final_shape = list(adv_shape) + list(leftover_dims)
    result = picked.view(*final_shape)
    return result


class TensorIndexingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return transform(x, [[[0,2],[1,3]]])
        
torch_model = TensorIndexingModel().eval()

raw_data = np.random.rand(5,5,5).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)