from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor# , WhispterConfig
import torch
from datasets import load_dataset


from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig



model_encoder = WhisperEncoder(WhisperConfig())





# get torch output

from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
# encoder = model.get_encoder()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

torch_model = model_encoder

raw_data = input_features.cpu().numpy()
torch_data = torch.from_numpy(raw_data)
pytorch_out = torch_model(torch_data)[0].detach().numpy()
print(pytorch_out)