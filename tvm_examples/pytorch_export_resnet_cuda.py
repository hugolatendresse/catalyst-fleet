"""
This is the example on https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html as-is!
Model Type: ResNet
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Result: FAIL (error from tvm: AssertionError: Unsupported function type batch_norm.default)
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build


import os
import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Give an example argument to torch.export
example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True)

mod, params = relax.frontend.detach_params(mod)
mod.show()

TOTAL_TRIALS = 1  # Change to 20000 for better performance if needed
target = tvm.target.Target("nvidia/geforce-rtx-4090")  # Change to your target device
work_dir = "tuning_logs"

# Skip running in CI environment
IS_IN_CI = os.getenv("CI", "") == "true"
if not IS_IN_CI:
    mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)

    # Only show the main function
    mod["main"].show()


if not IS_IN_CI:
    ex = relax.build(mod, target="cuda")
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)
    # Need to allocate data and params on GPU device
    gpu_data = tvm.nd.array(np.random.rand(1, 3, 224, 224).astype("float32"), dev)
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params).numpy()

    print(gpu_out.shape)