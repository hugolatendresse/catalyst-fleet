"""
Model Type: CNN
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Result: FAIL
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
import tvm
from tvm import relax
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
import torch.nn.functional as F
import numpy as np


# Create a dummy model
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PyTorchCNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # self.drop = nn.Dropout2d(p=0.2) # TODO retrain without dropout?
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # # Ensure input is in the correct format (assumes already in NCHW if using PyTorch DataLoader)
        # if not isinstance(x, torch.Tensor):
        #     x = self.transformation(x).float() # Converts HWC -> CHW
        #     x = x.unsqueeze(0)  # Converts CHW -> NCHW
        #     x = Variable(x)

        # Forward pass through CNN layers
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x)) # used to be: x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        
        # Flatten the tensor before passing to the fully connected layer
        x = x.view(x.size(0), -1)  # Use x.size(0) to handle batch size dynamically
        x = self.fc(x)
        
        # Return log probabilities for classification
        return F.log_softmax(x, dim=1)


# Give an example argument to torch.export
example_args = (torch.randn(1, 3, 128, 128, device='cuda', dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(PyTorchCNN().to("cuda").eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
mod_from_torch.show()

# exec = relax.build(mod_from_torch, target="llvm")
# dev = tvm.cuda()
# vm = relax.VirtualMachine(exec, dev)

# raw_data = np.random.rand(1, 3, 128, 128).astype("float32") # TODO copy raw data syntax below
# data = tvm.nd.array(raw_data, dev)
# cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
# print(cpu_out)

ex = relax.build(mod_from_torch, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
gpu_data = tvm.nd.array(np.random.rand(1, 3, 128, 128).astype("float32"), dev)
gpu_params = [tvm.nd.array(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params).numpy()

print(gpu_out.shape)