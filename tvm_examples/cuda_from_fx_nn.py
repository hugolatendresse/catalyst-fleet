"""
Model Type: simple nn
Model Definition: PyTorch
Model Export: fx tracer
Model Ingestion: from_fx
Target: CUDA
Compile and Run Test: ???
Correctness Test: ???
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
# input_tensors = [
    # torch.as_tensor(np.random.randn(*shape).astype(dtype))
    # for shape, dtype in input_info
# ]
torch_data = torch.from_numpy(raw_data)

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)

mod: tvm.IRModule = from_fx(graph_module, input_info)

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod)
print(mod.script())

exec = relax.build(mod_from_torch, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

tvm_input = tvm.nd.array(raw_data, dev)
tvm_out = vm["main"](tvm_input).numpy()
pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(tvm_out, pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 