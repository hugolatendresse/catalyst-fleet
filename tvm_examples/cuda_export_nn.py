"""
Model Type: simple NN
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Result: FAIL. TVM gives memory verification error.
"""
import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

# Create a dummy model
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Give an example argument to torch.export
example_args = (torch.randn(1, 784, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(TorchModel().eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
mod_from_torch.show()


from tvm import dlight as dl

with tvm.target.Target("cuda"):
    gpu_mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.LowBatchGEMV(),
        dl.gpu.Fallback(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Transpose(),
        dl.gpu.GeneralReduction(),
        dl.gpu.RMSNorm(),
    )(mod_from_torch)

exec = relax.build(gpu_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)
# Need to allocate data and params on GPU device
data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](data, *gpu_params).numpy()
print(gpu_out)
