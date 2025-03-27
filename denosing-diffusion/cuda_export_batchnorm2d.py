"""
Model Type: NN with batchnorm1D
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
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

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model)