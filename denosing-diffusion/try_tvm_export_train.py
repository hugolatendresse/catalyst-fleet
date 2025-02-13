"""
Model Type: train diffusion model
Model Definition: ?
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: LLVM
Compile and Run Test: FAIL: unable to export (pytorch error) 
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

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 10    # number of steps
)
from torch import nn
assert isinstance(diffusion, nn.Module), "Our methodology expects to ingest an nn.Module"


# Give an example argument to torch.export
example_args = (torch.randn(1, 3, 128, 128, dtype=torch.float32),)

# Convert the model to IRModule
torch_model = diffusion
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
mod_from_torch.show()

mod = mod_from_torch
exec = relax.build(mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

raw_data = np.random.rand(1, 3, 128, 128).astype("float32")
data = tvm.nd.array(raw_data, dev)
tvm_out = vm["main"](data, *params_from_torch["main"]).numpy()
print(tvm_out)
pytorch_out = torch_model(torch.from_numpy(raw_data)).detach().numpy() 
print(pytorch_out)
np.testing.assert_allclose(tvm_out, pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 