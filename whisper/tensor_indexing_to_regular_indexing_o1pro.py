
import torch
import torch
import torch

def _parse_index_spec(index_spec):
    """
    Convert the user-supplied `index_spec` into a *tuple* of advanced indices,
    matching how PyTorch interprets t[x], t[x,y], etc.
    
    Examples from your question:
      - [[[0,2],[1,3]]] => a single advanced index (shape (2,2))
      - [[0,2],[1,3]]   => two advanced indices
    """
    if isinstance(index_spec, tuple):
        # Already a tuple => assume multiple indices
        return index_spec

    if isinstance(index_spec, list):
        if len(index_spec) > 1:
            # Heuristic: if we see e.g. [[0,2],[1,3]], interpret that
            # as two advanced indices, not one nested shape
            # (which is how PyTorch does t[[0,2],[1,3]]).
            # But if you had [[[0,2],[1,3]]], that's shape (1,2,2),
            # which is a single advanced index. 
            # So we just say: if top-level length>1 and each top-level entry
            # is "indexy", treat them as separate.
            # Otherwise treat it as a single advanced index.
            top_level_items_are_all_int_or_list = all(
                isinstance(x, (int, list, tuple)) for x in index_spec
            )
            if top_level_items_are_all_int_or_list:
                # interpret as multiple indices
                return tuple(index_spec)
            else:
                # interpret as a single advanced index
                return (index_spec,)
        else:
            # length==1 => definitely a single advanced index
            return (index_spec,)

    # If it's an integer, or anything else, treat as single advanced index:
    return (index_spec,)

def _get_broadcast_shape(list_of_tensors):
    """
    Compute the broadcasted shape among all tensors in list_of_tensors.
    We can do this by a dummy add that forces broadcast, capturing the final shape.
    """
    shape = None
    for t in list_of_tensors:
        if shape is None:
            shape = t.shape
        else:
            # This forces a broadcast
            dummy = torch.zeros(shape, dtype=torch.long, device=t.device)
            out = dummy + t  # If shapes are incompatible, raises an error
            shape = out.shape
    return shape


def transform(t: torch.Tensor, index_spec):
    """
    Replicate what `t[index_spec]` does, using only:
     - regular slicing/view
     - broadcasting (expand)
     - torch.gather
    No reliance on built-in advanced indexing or index_select.
    
    Works for nested advanced indices such as [[[0,2],[1,3]]].
    """
    # 1) Convert user index into a tuple of advanced indices
    parsed = _parse_index_spec(index_spec)   # tuple of length m
    m = len(parsed)                          # number of advanced indices
    n = t.dim()                              # dimensionality of t

    # 2) Convert each advanced index into a LongTensor
    adv_indices = []
    for idx in parsed:
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=t.device)
        adv_indices.append(idx_t)

    # 3) Broadcast the advanced indices among themselves
    broadcast_shape = _get_broadcast_shape(adv_indices)  # shape B

    # leftover dimensions = dims that are not advanced
    leftover_dims = t.shape[m:]       # sizes for dims m..n-1
    out_shape = broadcast_shape + leftover_dims  # final shape

    # 4) Build coordinates for each dimension
    coords_for_dim = [None] * n

    # For the advanced dims (d < m), each adv_indices[d] is shape = broadcast_shape.
    # But we want them to tile across leftover_dims as well, so we expand to out_shape.
    # To do that, we must first .view() to (broadcast_shape + [1]*(# leftover dims)),
    # then .expand(out_shape).

    for d in range(m):
        idx_d = adv_indices[d]

        # idx_d is shape broadcast_shape, e.g. (1,2,2) in your example
        # We want it to become shape out_shape, e.g. (1,2,2,5,5).

        # Step A: view => shape B + (1,...,1) for leftover_dims
        view_shape = list(broadcast_shape) + [1] * (n - m)
        idx_d_viewed = idx_d.view(view_shape)  # insert trailing 1-dims

        # Step B: expand to out_shape
        idx_d_expanded = idx_d_viewed.expand(out_shape)

        coords_for_dim[d] = idx_d_expanded

    # For leftover dims (d >= m), we want the coordinate to range 0..(size-1).
    # Then we also replicate across the broadcast_shape *and* all leftover dims
    # except the one for dimension d.  We can do a systematic "view + expand" trick.
    for d in range(m, n):
        size_d = t.shape[d]  # leftover dim size
        # out_shape = broadcast_shape + leftover_dims
        # The position of this leftover dim 'd' within leftover_dims is (d - m).
        leftover_axis = (d - m)
        # That axis in the final out_shape is len(broadcast_shape) + leftover_axis

        # We'll build range_d: shape (size_d,)
        range_d = torch.arange(size_d, dtype=torch.long, device=t.device)

        # We'll then insert enough unsqueezes so that dimension
        # (len(broadcast_shape) + leftover_axis) is the one that goes 0..size_d-1
        num_out_dims = len(out_shape)
        vary_dim = len(broadcast_shape) + leftover_axis

        view_shape = [1] * num_out_dims
        view_shape[vary_dim] = size_d  # we want to vary along that dimension

        range_d_view = range_d.view(view_shape)   # shape [..., size_d, ...]
        range_d_expanded = range_d_view.expand(out_shape)

        coords_for_dim[d] = range_d_expanded

    # 5) Convert multi-dim coords to a single “flat” index
    #    for t.view(-1)
    # linear_index = sum_{k=0..n-1} coords_for_dim[k] * product_of_sizes(k+1..n-1)
    # We'll precompute those "multipliers" from right to left.
    shape_t = t.shape
    multipliers = [1]*n
    running = 1
    for d in reversed(range(n)):
        multipliers[d] = running
        running *= shape_t[d]

    linear_index = torch.zeros(out_shape, dtype=torch.long, device=t.device)
    for d in range(n):
        linear_index += coords_for_dim[d] * multipliers[d]

    # 6) gather from t.flat
    t_flat = t.view(-1)
    linear_index_flat = linear_index.view(-1)
    gathered_flat = torch.gather(t_flat, 0, linear_index_flat)
    result = gathered_flat.view(out_shape)

    return result


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