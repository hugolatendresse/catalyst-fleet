"""
Model Type: only batchnorm1D
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: FAIL
"""
import sys
sys.path.append('/ssd1/htalendr/tvm/python')
import numpy as np
from torch import nn
from hlutils.test_export_and_cuda import test_export_and_cuda
from hlutils.set_seed_all import set_seed_all

set_seed_all()

torch_model = nn.BatchNorm2d(2).eval()

raw_data = np.random.rand(1,2,2, 2).astype("float32")

test_export_and_cuda(raw_data, torch_model)