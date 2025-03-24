"""
Model Type: torch model with multiple arguments to forward 
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
import torch.nn.functional as F
import numpy as np
from typing import Optional


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y:
            z = x + y
        else:
            z = x
        
        return z

torch_model = TorchModel().eval()
raw_data = np.random.rand(1, 3, 128, 128).astype("float32")
torch_data = torch.from_numpy(raw_data)
example_args = (torch_data, None)

with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True
    )

tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

tvm_mod.show()

target = tvm.target.Target.from_device(tvm.cuda())

ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

gpu_data1 = tvm.nd.array(raw_data, dev)
gpu_data2 = None
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
print("About to run vm...")
gpu_out = vm["main"](gpu_data1, gpu_data2, *gpu_params)
print("Done! output is", gpu_out[0].numpy())

