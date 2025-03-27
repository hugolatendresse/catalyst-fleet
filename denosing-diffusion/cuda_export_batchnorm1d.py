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

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__() 
        self.fc1 = nn.Linear(784, 256) 
        self.bn1 = nn.BatchNorm1d(256)  
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.fc1(x) # 10, 256
        x = self.bn1(x) # 10, 256 
        x = self.relu1(x) # 10, 256
        x = self.fc2(x) # 10, 10
        return x
    
torch_model = TorchModel().eval()

raw_data = np.random.rand(10, 784).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)