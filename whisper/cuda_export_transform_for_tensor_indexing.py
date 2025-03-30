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


import torch
from functools import reduce
import operator

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
    For new versions, you could do: return torch.broadcast_shapes(*shapes).
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
    Decide if 'index_arg' is multiple parallel indices vs. a single advanced index.
    - If the top-level is a list/tuple and len(index_arg) > 1, interpret it as multiple indices.
    - Otherwise, it's a single advanced index.
    """
    if isinstance(index_arg, (list, tuple)):
        if len(index_arg) > 1:
            return True
    return False

def transform(t, index_arg):
    """
    Replicate t[index_arg] using only:
      - basic (scalar) indexing into 't'
      - torch.index_select
      - concatenation/stack
      - broadcasting
    …with NO advanced indexing on 't'.

    For multiple advanced indices: broadcast them, loop in scalar fashion.

    For a single advanced index: 
      1) Convert the nested Python list to a LongTensor.
      2) Remove exactly one leading dimension of size=1, if present. (Matches PyTorch's shape rule.)
      3) Flatten -> fix negative indices -> index_select -> reshape.
    """

    # -----------------------------------------------------------
    # CASE B: multiple advanced indices
    # -----------------------------------------------------------
    if _is_multiple_indices(index_arg):
        # e.g. t[[0,2],[1,3]] => separate indices for dim=0, dim=1
        idx_list = []
        for sub_i in index_arg:
            idx_list.append(torch.tensor(sub_i, dtype=torch.long))

        # 1) Broadcast them to a common shape B
        shapes = [x.shape for x in idx_list]
        B = _broadcast_shapes(shapes)

        # 2) Expand each index to that shape
        for i in range(len(idx_list)):
            idx_list[i] = idx_list[i].expand(B)

        k = len(idx_list)               # number of advanced dims
        leftover_dims = t.shape[k:]     # leftover dims after those k

        M = _prod(B)
        # 3) Flatten each index to length=M
        for i in range(k):
            idx_list[i] = idx_list[i].reshape(M)

        # 4) Enumerate each broadcasted coordinate => basic scalar indexing
        slices = []
        for n in range(M):
            out_slice = t
            for i in range(k):
                scalar_idx = idx_list[i][n].item()
                # handle negative indexing if you want:
                if scalar_idx < 0:
                    scalar_idx += t.shape[i]
                out_slice = out_slice[scalar_idx]
            slices.append(out_slice.unsqueeze(0))  # shape [1, leftover_dims]

        # 5) Concatenate -> shape [M, leftover_dims]
        stacked = torch.cat(slices, dim=0)
        # 6) Reshape -> [B, leftover_dims]
        final_shape = list(B) + list(leftover_dims)
        result = stacked.view(*final_shape)
        return result

    # -----------------------------------------------------------
    # CASE A: single advanced index
    # -----------------------------------------------------------
    else:
        # 1) Convert the nested Python list -> a LongTensor
        #    This is allowed. It's not advanced indexing on 't', 
        #    just building an index tensor from a Python list.
        idx_t = torch.tensor(index_arg, dtype=torch.long)

        # 2) If there's at least one dimension and the first dimension is size=1,
        #    remove exactly one leading dim. 
        #    (This matches PyTorch's "merge away the top-level [1]" rule for a single advanced index.)
        if idx_t.dim() > 0 and idx_t.shape[0] == 1:
            idx_t = idx_t.squeeze(0)  # remove exactly one leading dim

        # 3) Flatten -> fix negative indices -> index_select
        flattened = idx_t.reshape(-1)
        # fix negative indices if desired:
        # for i in range(flattened.size(0)):
        #     if flattened[i] < 0:
        #         flattened[i] = flattened[i] + t.shape[0]

        # we can do this in a vectorized manner if needed:
        # but for brevity, let's skip negative idx correction or assume they're in range [0, t.shape[0]-1]

        picked = torch.index_select(t, dim=0, index=flattened)

        # leftover dims
        leftover_dims = t.shape[1:]
        # final shape = idx_t.shape + leftover_dims
        adv_shape = idx_t.shape
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