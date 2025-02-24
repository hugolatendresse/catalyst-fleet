"""
Model Type: softmax op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: FAIL
Correctness Test: FAIL
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
from torch import nn
import numpy as np

from torch.nn import Softmax

torch_model = Softmax(dim=2).eval()

raw_data = np.random.rand(1, 4, 32, 8192).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model) 