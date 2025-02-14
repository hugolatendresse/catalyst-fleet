"""
Model Type: NN with batchnorm1D
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: FAIL (can't handle batchnorm.default)
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
        self.bn1 = nn.BatchNorm2d(12)  # BatchNorm after conv1
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)  # BatchNorm after conv2
        
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)  # BatchNorm after conv3
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Conv1 -> BatchNorm -> Pool -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # Conv2 -> BatchNorm -> Pool -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # Conv3 -> BatchNorm -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Dropout (if needed)
        x = F.dropout(x, training=self.training)
        
        # Flatten before the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

torch_model = PyTorchCNN().eval()

raw_data = np.random.rand(1, 3, 128, 128).astype("float32")
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


ex = relax.build(tvm_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)

gpu_data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params)

pytorch_out = torch_model(torch_data).detach().numpy() 
np.testing.assert_allclose(gpu_out[0].numpy(), pytorch_out, rtol=1e-5, atol=1e-5) 
print("Correctness test passed!") 
