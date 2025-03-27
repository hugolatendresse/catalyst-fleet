"""
Model Type: simple nn
Model Definition: PyTorch
Model Export: fx tracer
Model Ingestion: from_fx
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax

import numpy as np
import torch
from torch import fx
import tvm
import tvm.testing
from tvm.relax.frontend.torch import from_fx

import torch
from torch import _dynamo as dynamo

# Define the module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

    def forward(self, input):
        return self.linear(input)

# Instantiate the model and create the input info dict.
torch_model = MyModule().eval()

raw_data = np.random.rand(128,10).astype("float32")
input_info = [((128, 10), "float32")]


from hlutils import test_fx_and_cuda
test_fx_and_cuda(raw_data, torch_model, input_info)