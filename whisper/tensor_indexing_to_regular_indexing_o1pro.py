
import torch
import itertools


import torch

def _is_list_of_same_length_sublists(pyobj):
    """
    Returns True if pyobj is a list, length > 1, and
    every element is itself a list/tuple of the same length.
    Example: [[0,2],[1,3]] => True
             [[1,5]] => False (length=1)
             [[[0,2],[1,3]]] => False (top-level length=1)
    """
    if not isinstance(pyobj, list):
        return False
    if len(pyobj) <= 1:
        return False
    
    # Check that each element is a list (or tuple) of the same length
    first_len = None
    for elem in pyobj:
        if not isinstance(elem, (list, tuple)):
            return False
        length = len(elem)
        if first_len is None:
            first_len = length
        elif length != first_len:
            return False
    return True


def _as_long_tensor(pyobj):
    """Convert a nested list of ints to a torch.LongTensor."""
    return torch.tensor(pyobj, dtype=torch.long)


def _compute_linear_indices(shape_t, list_of_indices):
    """
    shape_t: tuple of t's dims, e.g. (D0, D1, D2, ...)
    list_of_indices: [idx0, idx1, ...] each is a LongTensor, broadcastable to same shape.
                     We assume we are indexing dims [0, 1, ..., len(list_of_indices)-1].
    
    Returns:
      lin_indices (LongTensor) of shape == broadcasted shape of all idx's.
      leftover_shape (tuple) for the dims that are not advanced-indexed.
      Also returns the broadcasted shape of the indices.
    """
    import math
    
    # shape_t[:k] are the dims we advanced-index, where k = len(list_of_indices)
    # leftover dims = shape_t[k:]
    # We must broadcast all idx_i to a common shape B.
    
    k = len(list_of_indices)
    leftover_shape = shape_t[k:]
    
    # First convert each index to a LongTensor.
    idx_tensors = []
    for i in range(k):
        idx_tensors.append(list_of_indices[i].long())

    # Broadcast them all to the same shape:
    # easiest is to do something like 
    #   bcasted = torch.broadcast_tensors(*idx_tensors)
    # which returns a list of Tensors each with same shape
    bcasted = torch.broadcast_tensors(*idx_tensors)
    
    # Now compute the linear index = idx0 * (D1*D2*...) + idx1*(D2*...) + ...
    # step by step
    dims_product = [1]*(k+1) 
    # dims_product[i] will store product of shape_t[i+1..k-1] (the dims between i and k)
    # Actually more precisely, for index i we multiply by product of shape_t[i+1..k-1], ignoring leftover dims for the moment.
    # e.g. if shape_t = (D0, D1, D2, D3, ...), indexing first 2 dims => k=2:
    #   linear_index = idx0*(D1*D2*D3...) + idx1*(D2*D3...) if we were indexing dims 0,1 fully to the end.
    # But PyTorch advanced indexing stops at dimension k for leftover. So the factor for i is product(D_{i+1}..D_{k-1}) * product_of_all leftover dims? 
    # Actually the standard formula for multi-dim indexing on the first k dims is:
    #   linear_idx = idx0 * (D1*D2*...*D_{n-1}) + idx1 * (D2*...*D_{n-1}) + ...
    # i.e. we flatten as if the shape is (D0, D1, ..., D_{n-1}) in row-major order.
    # So let's compute the full row-major strides for shape_t:
    
    strides = [1] * len(shape_t)
    # strides for row-major (Python) is reversed typically, let's do direct:
    # strides[-1] = 1
    for s in range(len(shape_t)-2, -1, -1):
        strides[s] = strides[s+1] * shape_t[s+1]

    # Now for dimension i in [0..k-1], the multiplier is strides[i].
    broadcast_shape = bcasted[0].shape  # all bcasted have the same shape
    linear_idx = torch.zeros(broadcast_shape, dtype=torch.long)
    for i in range(k):
        linear_idx += bcasted[i] * strides[i]
    
    return linear_idx, leftover_shape, broadcast_shape


def transform(t, some_list_of_lists):
    """
    Replicates t[some_list_of_lists] using only gather (plus .view, .expand, etc.).
    Handles both the 'multiple advanced index' case and the 'single advanced index' case.
    """
    # 1) Decide if we have multiple advanced indices or just one.
    if _is_list_of_same_length_sublists(some_list_of_lists):
        # => PyTorch interprets it as multiple advanced indices,
        # each indexing a different dimension in order.
        # e.g. [[0,2], [1,3]] => two advanced indices: dimension0 gets [0,2], dimension1 gets [1,3].
        
        # Convert each sub-list to a tensor:
        list_of_idxs = []
        for elem in some_list_of_lists:
            list_of_idxs.append(_as_long_tensor(elem))
        k = len(list_of_idxs)  # number of advanced dims
        # We will do the "flatten the entire tensor" approach (for correctness with diagonal picks),
        # compute linear indices, gather, reshape.
        
        shape_t = t.shape
        # compute linear index:
        lin_idx, leftover_shape, bshape = _compute_linear_indices(shape_t, list_of_idxs)
        
        # Flatten t fully to 1D:
        t_flat = t.view(-1)  # shape (D0*D1*...*D_{n-1},)
        
        # Now gather from t_flat along dim=0 with lin_idx
        # but we must flatten lin_idx as well:
        lin_idx_flat = lin_idx.view(-1)
        
        gathered_flat = torch.gather(t_flat, 0, lin_idx_flat)
        # shape => (lin_idx_flat.numel(),)
        
        # Finally reshape to bshape + leftover_shape
        # leftover_shape is a tuple, we want them all at the end:
        final_shape = bshape + leftover_shape
        out = gathered_flat.view(*final_shape)
        return out
    
    else:
        # => Single advanced index scenario, which PyTorch applies to dimension 0 by default.
        # e.g. some_list_of_lists = [[0,2],[1,3]] if it had an extra bracket around it
        # or some_list_of_lists = [0,2,4]
        # We'll do the dimension=0 gather trick.
        
        idx_t = _as_long_tensor(some_list_of_lists)  # entire nested list as one index
        # NOTE: If you want to "ignore" a leading dimension=1 as the user’s example #1 might do,
        # you can do a quick squeeze if the top-level is length=1.  Something like:
        # if isinstance(some_list_of_lists, list) and len(some_list_of_lists) == 1:
        #     # Attempt to remove that outer dimension
        #     idx_t = idx_t.squeeze(0)
        
        old_shape = t.shape
        D0 = old_shape[0]
        
        # flatten t to (D0, -1)
        t_flat = t.view(D0, -1)
        
        # flatten idx
        idx_flat = idx_t.view(-1)
        
        # gather
        expanded_index = idx_flat.unsqueeze(1).expand(idx_flat.shape[0], t_flat.shape[1])
        gathered_flat = torch.gather(t_flat, 0, expanded_index)
        
        # reshape back
        out = gathered_flat.view(*idx_t.shape, *old_shape[1:])
        return out




t = torch.randn(5, 5, 5)

inputs = (
    [[[0,2],[1,3]]] ,# correct is torch.Size([2, 2, 5, 5])
    [[0,2],[1,3]] ,# correct is torch.Size([2, 5]) 
    [[1,5]] ,
    [[0]] ,
    [[[1,2,5]]] ,
)

for some_list in inputs:
    print("\nUsing the following index:", some_list)

    torch_output = t[some_list]
    print(f"torch_output with that index: {torch_output.shape}")

    new_result = transform(t, some_list)
    print(f"your output: {new_result.shape}")
    assert torch.equal(torch_output, new_result), f"FAILED!\n pytorch: \n{torch_output}, us: \n{new_result}"
    print("PASSED!")