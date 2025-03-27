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
from typing import Optional, Union


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: bool):
        if y:
            return 2*x
        else:
            return 3*x

torch_model = TorchModel().eval()
raw_data1 = np.ones((2,2)).astype("float32")
raw_data2 = np.ones((2,2)).astype("float32")
torch_data1 = torch.from_numpy(raw_data1)
torch_data2 = torch.from_numpy(raw_data2)
example_args = (torch_data1, False)

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

gpu_data1 = tvm.nd.array(raw_data1, dev)
gpu_data2 = tvm.nd.array(raw_data2, dev)
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
print(tvm_params)
print("About to run vm...")
gpu_out = vm["main"](gpu_data1, gpu_data2, *gpu_params)
print("Done! output is", gpu_out[0].numpy())

