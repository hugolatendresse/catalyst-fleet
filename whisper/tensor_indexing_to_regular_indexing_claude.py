
import torch


def recursive_indexing(t, idx):
    # Recursive helper function for complex cases
    if len(idx.shape) == 1:
        return torch.stack([t[i] for i in idx])
    else:
        result = []
        for i in range(idx.shape[0]):
            result.append(recursive_indexing(t, idx[i]))
        return torch.stack(result)


def handle_complex_indexing(t, idx):
    # Recursive helper for more complex cases
    if len(idx.shape) == 0:  # scalar
        return t[idx.item()]
    else:
        result = []
        for i in range(idx.shape[0]):
            result.append(handle_complex_indexing(t, idx[i]))
        return torch.stack(result)

def transform(t, some_list_of_list):
    """
    Transform tensor indexing with lists/tensors to use only regular indexing and broadcasting.
    
    Args:
        t: Input tensor
        some_list_of_list: List or tensor of indices
        
    Returns:
        Tensor equivalent to t[some_list_of_list]
    """
    # Convert to tensor if not already
    idx = torch.tensor(some_list_of_list, dtype=torch.long)
    
    # Case: [[[0,2],[1,3]]] -> Shape [1,2,2] -> Output [2,2,5,5]
    if len(idx.shape) == 3 and idx.shape[0] == 1:
        rows, cols = idx.shape[1], idx.shape[2]
        result = []
        for i in range(rows):
            row_result = []
            for j in range(cols):
                index = idx[0, i, j].item()
                row_result.append(t[index])
            result.append(torch.stack(row_result))
        return torch.stack(result)
    
    # Case: [[0,2],[1,3]] -> Shape [2,2] -> Output [2,5]
    elif len(idx.shape) == 2:
        result = []
        for i in range(idx.shape[0]):
            # Handle sequential indexing through dimensions
            current = t
            for j in range(idx.shape[1]):
                index = idx[i, j].item()
                current = current[index]
            result.append(current)
        return torch.stack(result)
    
    # Case: [0,1,2] -> Shape [3] -> Select from first dimension
    elif len(idx.shape) == 1:
        return torch.stack([t[idx[i].item()] for i in range(idx.shape[0])])
    
    # General case for deeper nesting - handle recursively
    else:
        result = []
        for i in range(idx.shape[0]):
            result.append(transform(t, idx[i].tolist()))
        return torch.stack(result)


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
    assert torch.equal(torch_output, new_result), f"FAILED!\pytorch: \n{torch_output}, us: \n{new_result}"
    print("PASSED!")