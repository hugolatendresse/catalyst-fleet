"""
Model Type: ResNnet18 
Model Definition: torchvision import 
Model Export: fx tracer
Model Ingestion: from_fx
Target: LLVM
Compile and Run Test: SUCCESS
Correctness Test: FAIL
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

from torchvision.models.resnet import ResNet18_Weights, resnet18
torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

input_dims = (1, 3, 4, 4) # TODO decide if revert to 1,3,224,224 or keep

raw_data = np.random.rand(*input_dims).astype("float32")

input_info = [(input_dims, "float32")]

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)

# Use the dynamo.export() to export the PyTorch model to FX.
# graph_module = dynamo.export(torch_model, *input_tensors).graph_module

# Use the importer to import the PyTorch model to Relax.
mod: tvm.IRModule = from_fx(graph_module, input_info)

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod)
print(mod.script())

exec = relax.build(mod_from_torch, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

data = tvm.nd.array(raw_data, dev)
tvm_out = vm["main"](data).numpy()
print(tvm_out)
pytorch_out = torch_model(torch.from_numpy(raw_data)).detach().numpy() 
print(pytorch_out)
np.testing.assert_allclose(tvm_out, pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 