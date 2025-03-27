import torch 

t = torch.randn(5, 4, 3)
some_list = [0,2]

def transform(t, some_list):
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
    
    # Return the indexed tensor
    return t[processed_list]

tensor_args = torch.tensor(some_list)

assert torch.equal(t[tensor_args], transform(t, some_list.copy())), "FAILED!"
print("PASSED!")