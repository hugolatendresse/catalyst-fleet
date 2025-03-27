"""
Model Type: Whisper Encoder
Model Definition: PyTorch
Model Export: -
Model Ingestion: -
Target: -
Compile and Run Test: PASS
Correctness Test: -
"""

from transformers import WhisperProcessor
import torch
from datasets import load_dataset
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig

# Define model to ingest
torch_model = WhisperEncoder(WhisperConfig())

# Define data to ingest 
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
raw_data = input_features.cpu().numpy()
torch_data = torch.from_numpy(raw_data)

pytorch_out = torch_model(torch_data)[0].detach().numpy()
print(pytorch_out)