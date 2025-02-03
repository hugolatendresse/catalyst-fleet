
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

# Define the module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

    def forward(self, input):
        return self.linear(input)

# Instantiate the model and create the input info dict.
# torch_model = MyModule()
# input_info = [((128, 10), "float32")]
input_info = [((1, 3, 224, 224), "float32")]

example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# input_tensors = [
#     torch.as_tensor(np.random.randn(*shape).astype(dtype))
#     for shape, dtype in input_info
# ]

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)



fx.symbolic_trace(torch_model).graph.print_tabular()


irmodule = from_fx(graph_module, input_info)

print(mod)