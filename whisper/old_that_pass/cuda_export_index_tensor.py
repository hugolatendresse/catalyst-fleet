"""
Model Type: 
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
"""
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import torch.nn.functional as F
import numpy as np

# Create a dummy model
class IndexTensorModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[torch.tensor([0,2])]
        
torch_model = IndexTensorModel().eval()

raw_data = np.random.rand(3,4).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda


torch_data = torch.from_numpy(raw_data)

# Give an example argument to torch.export
example_args = (torch_data,)
debug = True
# Convert the model to IRModule
# TODO what does , unwrap_unit_return_tuple=True do? should we include?
with torch.no_grad():
    if debug:
        print("Exporting model...")
    exported_program = export(torch_model, example_args)
    if debug:
        print("Converting model to IRModule...")
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
    )

if debug:
    print("Detaching parameters...")
tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
tvm_mod.show()

target = tvm.target.Target.from_device(tvm.cuda())

if debug:
    print("Defining VM...")
ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

if debug:
    print("Running VM...")
gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)

if debug:
    print("Running PyTorch model...")
pytorch_out = torch_model(torch_data)

if debug:
    print("Comparing outputs...")
if isinstance(pytorch_out, tuple):
    for i in range(len(pytorch_out)):
        actual = gpu_out[i].numpy()
        desired = pytorch_out[i].detach().numpy()
        np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)
else:
    actual = gpu_out[0].numpy()
    desired = pytorch_out.detach().numpy()
    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)

print("Correctness test passed!") 
