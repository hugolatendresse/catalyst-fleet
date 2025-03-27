"""
Model Type: split op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: FAIL
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
from torch import nn


# batch = 1
# Works with channel = 4 but not with channel = 5
channels = 11 # TODO try 139
# width = 1

chunks = 3 # TODO try 42
dim = 0

raw_data = np.random.rand(channels).astype("float32")

class SplitModelSplitSize(nn.Module):
    def __init__(self, split_size, dim):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, split_size_or_sections=self.split_size, dim=self.dim)

import math
split_size = math.ceil(raw_data.shape[dim] / chunks)
print("split size is", split_size) # split_size = 6
# torch_output[0] has shape torch.Size([6])
# torch_output[1] has shape torch.Size([5])
torch_model = SplitModelSplitSize(split_size=4, dim=dim).eval()

torch_data = torch.from_numpy(raw_data)

torch_output = torch_model(torch_data)
print("torch_output", torch_output)

print("torch_output has length", len(torch_output))
for i,x in enumerate(torch_output):
    print("torch_output[{}] has shape".format(i), x.shape)

# Give an example argument to torch.export
example_args = (torch_data,)

# Convert the model to IRModule
# TODO what does , unwrap_unit_return_tuple=True do? should we include?
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
    )

tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

target = tvm.target.Target.from_device(tvm.cuda())

ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)


print("chunk torch_output has length", len(torch_output))
assert len(torch_output) == len(gpu_out), f"different lengths!!. torch: {len(torch_output)}, gpu_out: {len(gpu_out)}"

print("the ouptut on both sides has the same length: ", len(torch_output))

print("torch_output[0].shape", torch_output[0].shape)
print("gpu_out[0].shape", gpu_out[0].shape)
print("torch_output[1].shape", torch_output[1].shape)
print("gpu_out[1].shape", gpu_out[1].shape)
print("torch_output[2].shape", torch_output[2].shape)
print("gpu_out[2].shape", gpu_out[2].shape)

print(type(torch_output))
print(type(gpu_out))
for i in range(len(torch_output)):
    print(f"iteration {i} of outputs have shape torch_output[i].shape {torch_output[i].shape} and gpu_out[i].numpy().shape {gpu_out[i].numpy().shape}")
    assert torch_output[i].shape == gpu_out[i].numpy().shape, "different shapes!!"
    np.testing.assert_allclose(actual=torch_output[i].detach().numpy(), desired=gpu_out[i].numpy(), rtol=1e-5, atol=1e-5)
print("Correctness test passed!") 
