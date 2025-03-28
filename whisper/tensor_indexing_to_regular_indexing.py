import torch 

t = torch.randn(5, 5, 5)
some_list = [[[0,2],[1,3]]]

def transform(t, some_list):
    print("transforming with some_list:", some_list)

    # Handle None or empty list
    if some_list is None:
        raise TypeError("'NoneType' object is not iterable")
    
    if isinstance(some_list, list) and len(some_list) == 0:
        return t[[]]
    
    # Process boolean values and preserve list structure
    def process_list(lst):
        if not isinstance(lst, list):
            # Convert boolean values to integers (0 or 1)
            if isinstance(lst, bool):
                return 1 if lst else 0
            return lst
        
        # Recursively process nested lists
        return [process_list(item) for item in lst]
    
    # Process the input list
    processed_list = process_list(some_list)

    print("processed_list:", processed_list)
    out = t[processed_list]
    print(f"result with processed list (shape is {out.shape})", out)
    # Return the indexed tensor
    return out


tensor_args = some_list
torch_output = t[tensor_args]
print(f"torch_output with original list (shape is {torch_output.shape})", torch_output)

assert torch.equal(torch_output, transform(t, some_list.copy())), "FAILED!"
print("PASSED!")