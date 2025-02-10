"""
Model Type: simple NN
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: SUCCESS
Correctness Test: SUCCESS
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build


import os
import numpy as np
import torch
from torch.export import export
from torch import nn

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

torch_model = TorchModel().eval()

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Give an example argument to torch.export
example_args = (torch.randn(10,784, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True)

mod, params = relax.frontend.detach_params(mod)
mod.show()

# TOTAL_TRIALS = 1  # Change to 20000 for better performance if needed
# target = tvm.target.Target("nvidia/geforce-rtx-4090")  # Change to your target device
# work_dir = "tuning_logs"

# # Skip running in CI environment
# IS_IN_CI = os.getenv("CI", "") == "true"
# if not IS_IN_CI:
#     mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)

#     # Only show the main function
#     mod["main"].show()


from tvm import dlight as dl

mod = tvm.relax.transform.LegalizeOps()(mod)

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
    )(mod)

ex = relax.build(gpu_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
raw_data = np.random.rand(10, 784).astype("float32")
gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)
print(gpu_out[0].numpy())

try:
    print(gpu_out.numpy())
except:
    print("gpu_out.numpy() doesn't work!")

try:
    print(gpu_out.shape)
except:
    print("gpu_out.shape doesn't work!")

print("gpu_out.handle:", gpu_out.handle)
print("dir(gpu_out):", dir(gpu_out))


pytorch_out = torch_model(torch.from_numpy(raw_data)).detach().numpy() 
print(pytorch_out)
np.testing.assert_allclose(gpu_out[0].numpy(), pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 