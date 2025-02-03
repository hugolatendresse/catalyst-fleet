
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

import numpy as np
import requests
import torch
from PIL import Image
import torch

import numpy as np
import torch

# Model
torch_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

input_info = [((720, 1280, 3), "float32")]
# example_args = [torch.randn(720, 1280, 3, dtype=torch.float32),]
# imgs = np.random.rand(720, 1280, 3, dtype=torch.float32)

im = Image.open(requests.get('https://ultralytics.com/images/zidane.jpg', stream=True).raw)
im = np.asarray(im)
imgs = []
imgs.append(im)

with torch.no_grad():
    # print("PASSING THIS:")
    # print(imgs[0].shape)
    output = torch_model(im)

output.print()
output.save()

# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)


fx.symbolic_trace(torch_model).graph.print_tabular()


irmodule = from_fx(graph_module, input_info)
# print(irmodule)

rt_lib_target = tvm.build(irmodule, target="llvm") # TODO why doesn't this work?
tvm_input = tvm.nd.array(example_args[0])
out = rt_lib_target["main"](tvm_input)
print(out)