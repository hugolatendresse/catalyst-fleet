"""
Model Type: Upsample op only
Model Definition: PyTorch
Model Export: torch.export
Model Ingestion: tvm.relax.frontend.torch.from_exported_program
Target: CUDA
Compile and Run Test: PASS
Correctness Test: PASS
TVM Status: opened PR #17721
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build
import numpy as np
from torch.nn import Upsample
from hlutils.set_seed_all import set_seed_all
set_seed_all()


batch_size = 2  
channels = 3
height, width = 32, 32


torch_model = Upsample(size=None, scale_factor = 7, mode = 'nearest',align_corners=None, recompute_scale_factor=True)
# TODO try with including/excluding the different parameters set to None

raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")

from hlutils.test_export_and_cuda import test_export_and_cuda
test_export_and_cuda(raw_data, torch_model) 