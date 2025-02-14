"""
Model Type: simple unet
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: ??
Correctness Test: ??
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
from denoising_diffusion_pytorch import Unet

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

torch_model = model.eval()

raw_data = np.random.rand(1,3,128, 128).astype("float32")
torch_data = torch.from_numpy(raw_data)

# Give an example argument to torch.export
example_args = (torch_data, torch.tensor([1], dtype=torch.float32))

# Convert the model to IRModule
# TODO what does , unwrap_unit_return_tuple=True do? should we include?
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
    )

tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
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
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)

pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(gpu_out[0].numpy(), pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 