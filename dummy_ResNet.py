
# import os
# os.environ['PYTHONPATH'] = '/ssd1/htalendr/tvm/python:' + os.environ.get('PYTHONPATH', '')
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

from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

input_info = [((1, 3, 224, 224), "float32")]

example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

with torch.no_grad():
    output = torch_model(*example_args)

print(output)
print(output.shape)

# input_tensors = [
#     torch.as_tensor(np.random.randn(*shape).astype(dtype))
#     for shape, dtype in input_info
# ]

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)



fx.symbolic_trace(torch_model).graph.print_tabular()


irmodule = from_fx(graph_module, input_info)
# print(irmodule)

rt_lib_target = tvm.build(irmodule, target="llvm") # TODO why doesn't this work?
tvm_input = tvm.nd.array(example_args[0])
out = rt_lib_target["main"](tvm_input)
print(out)