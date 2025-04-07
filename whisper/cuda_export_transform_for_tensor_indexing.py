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

def _prod(shape):
    """Compute the product of all dimensions in 'shape'."""
    out = 1
    for s in shape:
        out *= s
    return out

def _broadcast_shapes(shapes):
    # equivalent to  `return torch.broadcast_shapes(*shapes)`, but can't find how to have broadcast_shapes in TVM. 
    # TODO Try to understand what exported translator does when I pass broadcast_shapes, since cuda_export_index_broadcat_shape
    """
    Re-implementation of torch.broadcast_shapes since not sure how to call torch.broadcast_shapes(*shapes) in topi
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
    

def _is_multiple_indices(indices):
    """
    Decide if 'indices' is multiple parallel indices vs. a single advanced index.
    - If the top-level is a list/tuple and len(indices) > 1, interpret it as multiple indices.
    - Otherwise, it's a single advanced index.
    """
    if isinstance(indices, (list, tuple)):
        if len(indices) > 1:
            return True
    return False

def transform_tensor_index(data, indices):
    """
    Replicate data[indices] using only:
    - basic indexing on data
    - torch.index_select
    - concatenation/stack
    - broadcasting
    … and no advanced indexing.

    Approach for multiple advanced indices: broadcast and loop

    Approach for single advanced index: 
    1. Convert the nested Python list to a LongTensor.
    2. Remove exactly one leading dimension of size=1, if present. (Matches PyTorch's shape rule.)
    3. Flatten -> fix negative indices -> index_select -> reshape.
    """

    # -----------------------------------------------------------
    # CASE B: multiple advanced indices
    # -----------------------------------------------------------
    if _is_multiple_indices(indices):
        # e.g. data[[0,2],[1,3]] => separate indices for dim=0, dim=1
        idx_list = []
        for sub_i in indices:
            idx_list.append(torch.tensor(sub_i, dtype=torch.long))

        # 1) Broadcast them to a common shape B
        shapes = [x.shape for x in idx_list]
        B = _broadcast_shapes(shapes) 
        
        # 2) Expand each index to that shape
        for i in range(len(idx_list)):
            idx_list[i] = idx_list[i].expand(B)

        k = len(idx_list)               # number of advanced dims
        leftover_dims = data.shape[k:]     # leftover dims after those k

        M = _prod(B)
        # 3) Flatten each index to length=M
        for i in range(k):
            idx_list[i] = idx_list[i].reshape(M)

        # 4) Enumerate each broadcasted coordinate => basic scalar indexing
        slices = []
        for n in range(M):
            out_slice = data
            for i in range(k):
                scalar_idx = idx_list[i][n].item()
                # handle negative indexing if you want:
                if scalar_idx < 0:
                    scalar_idx += data.shape[i]
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
        #    This is allowed. It's not advanced indexing on 'data', 
        #    just building an index tensor from a Python list.
        idx_t = torch.tensor(indices, dtype=torch.long)

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
        #         flattened[i] = flattened[i] + data.shape[0]

        # we can do this in a vectorized manner if needed:
        # but for brevity, let's skip negative idx correction or assume they're in range [0, data.shape[0]-1]

        picked = torch.index_select(data, dim=0, index=flattened)

        # leftover dims
        leftover_dims = data.shape[1:]
        # final shape = idx_t.shape + leftover_dims
        adv_shape = idx_t.shape
        final_shape = list(adv_shape) + list(leftover_dims)
        result = picked.view(*final_shape)
        return result



data = torch.randn(5, 5, 5)

inputs = (
    [[[0,2],[1,3]]],  # correct output has dimensions torch.Size([2, 2, 5, 5])
    [[0,2],[1,3]],  # correct output has dimensions torch.Size([2, 5]) 
    [[1,4]],  # correct output has dimensions torch.Size([2, 5, 5])
    [[0]],  # correct output has dimensions torch.Size([1, 5, 5])
    [[[1,2,4]]],  # correct output has dimensions torch.Size([1, 3, 5, 5])
    # TODO need to test with slicing too! Like index = torch.Tensor([0:2, 1:3])
)

for some_list in inputs:
    print("\nUsing the following index:", some_list)

    torch_output = data[some_list]
    print(f"torch_output with that index: {torch_output.shape}")

    new_result = transform_tensor_index(data, some_list)
    print(f"your output: {new_result.shape}")
    assert torch.equal(torch_output, new_result), f"FAILED!\npytorch: \n{torch_output}, \nus: \n{new_result}"
    print("PASSED!")

    

class TensorIndexingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return transform_tensor_index(x, [[[0,2],[1,3]]])
        
torch_model = TensorIndexingModel().eval()

raw_data = np.random.rand(5,5,5).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)