"""
Model Type: only conv2d
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ?
Correctness Test: ?
"""
import sys
sys.path.append('/ssd1/htalendr/tvm/python')
import numpy as np
from torch.nn import Conv2d
from hlutils.test_export_and_cuda import test_export_and_cuda
from hlutils.set_seed_all import set_seed_all

set_seed_all()

import torch
import torch.nn as nn

batch_size = 1
in_channels = 3  # RGB image
height, width = 28, 28

x = torch.randn(batch_size, in_channels, height, width)

conv_layer = nn.Conv2d(
    in_channels=in_channels,       
    out_channels=16,               
    kernel_size=3,                 
    stride=2,                      
    padding=1,                     
    dilation=2,                    
    groups=1,                      
    bias=True,                     
    padding_mode='zeros',          
    device=torch.device('cpu'),    
    dtype=torch.float32            
)


torch_model = conv_layer.eval()

raw_data = np.random.rand(batch_size, in_channels, height, width).astype("float32")

test_export_and_cuda(raw_data, torch_model)