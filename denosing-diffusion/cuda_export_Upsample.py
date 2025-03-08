"""
Model Type: Upsample op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: FAIL
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
import numpy as np
from torch.nn import Upsample

torch_model = Upsample(scale_factor = 2, mode = 'nearest')

raw_data = np.random.rand(1, 32, 2, 2).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda

test_export_and_cuda(raw_data, torch_model) 