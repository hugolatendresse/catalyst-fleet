"""
Model Type: ResNet18
Model Definition: torchvision import 
Model Export: fx tracer
Model Ingestion: from_fx
Target: CUDA
Compile and Run Test: FAIL  Check failed: (it != n->end()) is false: cannot find the corresponding key in the Map
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

mod: tvm.IRModule = from_fx(graph_module, input_info)

tvm_mod, tvm_params = relax.frontend.detach_params(mod)
tvm_mod.show()

from tvm import dlight as dl

tvm_mod = tvm.relax.transform.LegalizeOps()(tvm_mod)

with tvm.target.Target("cuda"):
    tvm_mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.LowBatchGEMV(),
        dl.gpu.Fallback(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Transpose(),
        dl.gpu.GeneralReduction(),
        dl.gpu.RMSNorm(),
    )(tvm_mod)

exec = relax.build(tvm_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)

gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = []
gpu_out = vm["main"](gpu_data, *gpu_params)

tvm_input = tvm.nd.array(raw_data, dev)
# TODO why is it ok to not pass gpu_params below?
# gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
tvm_out = vm["main"](tvm_input).numpy()
pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(tvm_out, pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 