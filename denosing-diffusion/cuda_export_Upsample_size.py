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
from hlutils.set_seed_all import set_seed_all
set_seed_all()

batch_size = 1
channels = 3
height, width = 8, 8

torch_model = Upsample(
    size=(64, 64),               
    scale_factor=None,           # Not used when size is specified
    mode='nearest',             
    recompute_scale_factor=None)


raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda
test_export_and_cuda(raw_data, torch_model) 