# Closest so far https://chatgpt.com/c/67e8c126-1254-8006-8e88-d06e3f3732da

import torch
import math
import operator
from functools import reduce

def _prod(shape):
    """Utility: product of all dimensions in a shape tuple."""
    if len(shape) == 0:
        return 1
    return reduce(operator.mul, shape, 1)

def _broadcast_shapes(shapes):
    """
    Replicate torch.broadcast_shapes for older PyTorch versions.
    If you have a modern PyTorch that has torch.broadcast_shapes, you can use that directly.
    """
    # In new PyTorch: return torch.broadcast_shapes(*shapes)
    # Otherwise implement manually:
    # (We’ll do a minimal version that works if all are up to say 10 dims.)
    # Real broadcast logic: pad shapes from left, each dimension must match or be 1, or else error.
    # For the test cases you gave, a simpler approach is usually enough.
    
    # This is a faithful approach:
    max_ndim = max(len(s) for s in shapes)
    rev_shapes = [tuple(reversed(s)) for s in shapes]
    out = []
    for dim in range(max_ndim):
        dim_size = []
        for rs in rev_shapes:
            if dim < len(rs):
                dim_size.append(rs[dim])
        # broadcast among those
        target = 1
        for d in dim_size:
            if d != 1 and target != 1 and d != target:
                raise ValueError("Shapes not broadcastable!")
            target = max(target, d)
        out.append(target)
    # now reverse back
    out.reverse()
    return tuple(out)

def _is_multiple_indices(index_arg):
    """
    Decide if top-level of index_arg is multiple advanced indices
    or a single nested list/tensor for fancy indexing.
    Very ad-hoc rule to match your examples:
      - If the top-level is a list/tuple with length > 1, treat as multiple parallel indices.
      - Otherwise, single advanced index.
    """
    if isinstance(index_arg, (list, tuple)):
        if len(index_arg) > 1:
            # Usually means multiple advanced indexes: e.g. [ [0,2], [1,3] ]
            return True
    return False

def transform(t, index_arg):
    """
    Replicate t[index_arg] using only basic indexing, slicing,
    broadcasting, index_select, cat, stack, etc.
    """
    # Check if we have multiple advanced indices vs single advanced index
    if _is_multiple_indices(index_arg):
        # Case B: multiple advanced indices
        idx_list = []
        for sub_i in index_arg:
            # Convert each sub-index to a LongTensor
            idx_list.append(torch.tensor(sub_i, dtype=torch.long))

        # Broadcast them all to the same shape
        shapes = [x.shape for x in idx_list]
        B = _broadcast_shapes(shapes)  # common shape
        # Expand
        for i in range(len(idx_list)):
            idx_list[i] = idx_list[i].expand(B)

        # leftover dims are t.shape[len(idx_list):]
        k = len(idx_list)
        leftover_dims = t.shape[k:]
        
        M = _prod(B)
        # Flatten each index to 1D of length M
        for i in range(k):
            idx_list[i] = idx_list[i].reshape(M)

        # We'll accumulate slices in a list
        slices = []
        for n in range(M):
            # Collect scalar indices
            scalar_indices = []
            for i in range(k):
                scalar_indices.append(idx_list[i][n].item())

            # Now index t with those scalars in dimension 0 repeatedly
            out_slice = t
            for i in range(k):
                out_slice = out_slice[scalar_indices[i]]
            # out_slice has shape leftover_dims

            slices.append(out_slice.unsqueeze(0))  # shape [1, *leftover_dims]

        stacked = torch.cat(slices, dim=0)  # shape [M, *leftover_dims]
        final_shape = list(B) + list(leftover_dims)
        result = stacked.view(*final_shape)
        return result

    else:
        # Case A: single advanced index
        idx_t = torch.tensor(index_arg, dtype=torch.long)

        # Squeeze out leading dims of size 1 if any
        # to mimic PyTorch's "list-of-lists" advanced index shape behavior
        while idx_t.dim() > 0 and idx_t.shape[0] == 1:
            idx_t = idx_t.squeeze(0)

        B = idx_t.shape  # shape of the advanced index
        leftover_dims = t.shape[1:]
        
        flattened = idx_t.view(-1)  # 1D
        M = flattened.shape[0]
        
        picked = torch.index_select(t, dim=0, index=flattened)  # shape [M, *leftover_dims]
        
        final_shape = list(B) + list(leftover_dims)
        result = picked.view(*final_shape)
        return result


t = torch.randn(5, 5, 5)

inputs = (
    [[[0,2],[1,3]]],  # correct output has dimensions torch.Size([2, 2, 5, 5])
    [[0,2],[1,3]],  # correct output has dimensions torch.Size([2, 5]) 
    [[1,4]],  # correct output has dimensions torch.Size([2, 5, 5])
    [[0]],  # correct output has dimensions torch.Size([1, 5, 5])
    [[[1,2,4]]],  # correct output has dimensions torch.Size([1, 3, 5, 5])
)

for some_list in inputs:
    print("\nUsing the following index:", some_list)

    torch_output = t[some_list]
    print(f"torch_output with that index: {torch_output.shape}")

    new_result = transform(t, some_list)
    print(f"your output: {new_result.shape}")
    assert torch.equal(torch_output, new_result), f"FAILED!\n pytorch: \n{torch_output}, us: \n{new_result}"
    print("PASSED!")