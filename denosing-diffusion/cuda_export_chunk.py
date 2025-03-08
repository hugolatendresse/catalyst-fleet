"""
Model Type: chunk op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ?
Correctness Test: ?
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

raw_data = np.random.rand(3, 4, 7, 11).astype("float32")

class ChunkModel(nn.Module):
    def __init__(self, chunks, dim):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x):
        return x.chunk(self.chunks, dim=self.dim)

torch_model = ChunkModel(chunks=2, dim=1).eval()

torch_data = torch.from_numpy(raw_data)

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

torch_output = torch_model(torch_data)
desired_0 = torch_output[0].detach().numpy()
desired_1 = torch_output[1].detach().numpy()
actual_0 = gpu_out[0].numpy()
actual_1 = gpu_out[1].numpy()
np.testing.assert_allclose(actual=actual_0, desired=desired_0, rtol=1e-5, atol=1e-5) 
np.testing.assert_allclose(actual=actual_1, desired=desired_1, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 

