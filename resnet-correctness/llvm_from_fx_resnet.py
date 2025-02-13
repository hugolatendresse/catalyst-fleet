"""
Model Type: ResNet18 
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
from torch import _dynamo as dynamo

from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

raw_data = np.random.rand(1, 3, 224, 224).astype("float32")
input_info = [((1, 3, 224, 224), "float32")]
torch_data = torch.from_numpy(raw_data)

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)

mod: tvm.IRModule = from_fx(graph_module, input_info)

tvm_mod, tvm_params = relax.frontend.detach_params(mod)
tvm_mod.show()

exec = relax.build(tvm_mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

tvm_input = tvm.nd.array(raw_data, dev)
tvm_out = vm["main"](tvm_input).numpy()
pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(tvm_out, pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 