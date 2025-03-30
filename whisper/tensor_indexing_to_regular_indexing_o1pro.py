# Closest so far https://chatgpt.com/c/67e8c126-1254-8006-8e88-d06e3f3732da

import torch
import operator
from functools import reduce

def _prod(shape):
    """Utility: product of all dimensions in a shape tuple."""
    out = 1
    for s in shape:
        out *= s
    return out

def _broadcast_shapes(shapes):
    """
    Minimal re-implementation of torch.broadcast_shapes for older PyTorch.
    For new PyTorch: could just do return torch.broadcast_shapes(*shapes).
    """
    max_ndim = max(len(s) for s in shapes)
    # We'll compare shapes from the right.
    rev_shapes = [s[::-1] for s in shapes]
    out = []
    for i in range(max_ndim):
        dim_size = 1
        for rs in rev_shapes:
            if i < len(rs):
                s_ = rs[i]
                if s_ != 1 and dim_size != 1 and s_ != dim_size:
                    raise ValueError(f"Incompatible shapes for broadcast: {shapes}")
                dim_size = max(dim_size, s_)
        out.append(dim_size)
    out.reverse()
    return tuple(out)

def _is_multiple_indices(index_arg):
    """
    Decide if index_arg should be interpreted as multiple parallel indices
    vs. a single advanced index. We replicate the usual PyTorch approach:
      - If the top-level is a list/tuple *and* len(index_arg) > 1,
        we interpret them as multiple indices.
      - Otherwise, it's a single advanced index.
    """
    if isinstance(index_arg, (list, tuple)):
        if len(index_arg) > 1:
            return True
    return False

def _parse_single_index(x):
    """
    Recursively 'squeeze' out leading singleton layers in a nested Python list
    whenever the current list/tuple has exactly one element.

    PyTorch does something equivalent when you do t[[[...]]] with a single advanced
    index: it merges away leading dimensions of size 1 in the nested Python list.

    For example:
      - [[[0,2],[1,3]]]  -> [[0,2],[1,3]]
      - [[0]]            -> [0]
      - [[[1,2,4]]]      -> [[1,2,4]]
      - [[1,4]]          -> [1,4]
    so that the final shape matches what PyTorch uses internally.
    """
    if not isinstance(x, (list, tuple)):
        # It's already a scalar
        return x

    # If there's exactly one element, descend into it
    if len(x) == 1:
        return _parse_single_index(x[0])
    else:
        # Otherwise, parse each child
        return [_parse_single_index(elem) for elem in x]

def transform(t, index_arg):
    """
    Replicate t[index_arg] using:
      - basic Python scalar indexing
      - slicing
      - torch.index_select
      - cat, stack
      - broadcasting
    without directly using advanced indexing on t.
    """
    # ------------------------------------------------------------------
    # Case B: multiple advanced indices
    # ------------------------------------------------------------------
    if _is_multiple_indices(index_arg):
        # We have something like [ [0,2], [1,3] ], meaning dimension0 = [0,2], dimension1=[1,3], etc.
        idx_list = []
        for sub_i in index_arg:
            idx_list.append(torch.tensor(sub_i, dtype=torch.long))

        # Broadcast them to a common shape
        shapes = [x.shape for x in idx_list]
        B = _broadcast_shapes(shapes)

        # Expand each to the broadcast shape
        for i in range(len(idx_list)):
            idx_list[i] = idx_list[i].expand(B)

        k = len(idx_list)                   # how many advanced dims
        leftover_dims = t.shape[k:]         # leftover shape after those k dims

        M = _prod(B)
        # Flatten them to length M
        for i in range(k):
            idx_list[i] = idx_list[i].reshape(M)

        # We'll collect slices (each is leftover_dims) by scalar indexing
        slices = []
        for n in range(M):
            out_slice = t
            for i in range(k):
                scalar_idx = idx_list[i][n].item()
                out_slice = out_slice[scalar_idx]
            # now out_slice has shape leftover_dims
            slices.append(out_slice.unsqueeze(0))  # [1, *leftover_dims]

        # stack -> [M, *leftover_dims]
        stacked = torch.cat(slices, dim=0)

        # reshape -> [*B, *leftover_dims]
        final_shape = list(B) + list(leftover_dims)
        result = stacked.view(*final_shape)
        return result

    # ------------------------------------------------------------------
    # Case A: single advanced index
    # ------------------------------------------------------------------
    # 1) Parse the nested list so that we remove repeated leading [1]-dim layers.
    parsed = _parse_single_index(index_arg)
    # 2) Convert to a tensor of type long
    idx_t = torch.tensor(parsed, dtype=torch.long)
    #    Now idx_t.shape should match how PyTorch interprets the shape of that advanced index.

    # 3) Flatten idx_t
    flat_idx = idx_t.view(-1)

    # 4) index_select along dim=0
    picked = torch.index_select(t, dim=0, index=flat_idx)  # shape [M, *t.shape[1:]], M=flat_idx.numel()

    # 5) Reshape to [*idx_t.shape, *t.shape[1:]]
    adv_shape = idx_t.shape
    leftover_dims = t.shape[1:]
    final_shape = list(adv_shape) + list(leftover_dims)
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
    assert torch.equal(torch_output, new_result), f"FAILED!\npytorch: \n{torch_output}, \nus: \n{new_result}"
    print("PASSED!")