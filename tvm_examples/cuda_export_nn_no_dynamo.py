"""
File: cuda_export_nn.py
Model Type: simple NN
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: FAIL
Correctness Test: FAIL
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

    # @torch._dynamo.disable
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


torch_model = TorchModel().eval()
compiled_model = torch.compile(torch_model, mode="reduce-overhead")



raw_data = np.random.rand(10, 784).astype("float32")
torch_data = torch.from_numpy(raw_data)

# Give an example argument to torch.export
example_args = (torch_data,)

# Convert the model to IRModule
# TODO what does , unwrap_unit_return_tuple=True do? should we include?

import os
print("PYDEVD_USE_FRAME_EVAL:", os.environ.get("PYDEVD_USE_FRAME_EVAL"))

with torch.no_grad():
    exported_program = export(compiled_model, example_args)
    mod_from_torch = from_exported_program(
            exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
        )

tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
tvm_mod.show()

target = tvm.target.Target.from_device(tvm.cuda())

ex = relax.build(tvm_mod, target=target, pipeline=relax.get_default_pipeline(target))
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)

pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(gpu_out[0].numpy(), pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 