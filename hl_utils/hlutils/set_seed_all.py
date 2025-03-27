import random
import numpy as np
import torch

def set_seed_all(seed=42) -> None:
    random.seed(seed)                   # Lock Python's random module
    np.random.seed(seed)                # Lock NumPy's random generator
    torch.manual_seed(seed)             # Lock PyTorch's CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    # Lock PyTorch's GPU seed for the current GPU
        torch.cuda.manual_seed_all(seed)  # Lock PyTorch's GPU seed for all GPUs (if using more than one)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA convolution algorithms
    