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
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
encoder = model.get_encoder()

assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features


class GenerateWrapper(torch.nn.Module):
    def __init__(self, model, assistant_model):
        super().__init__()
        self.model = model
        self.assistant_model = assistant_model
        
    def forward(self, input_features):
        # Instead of calling generate, implement the core computation
        # that you need for inference
        encoder_outputs = self.model.get_encoder()(input_features)
        return encoder_outputs[0]
        # # You might need to modify this part based on what you actually need
        # decoder_output = self.model.get_decoder()(
        #     input_ids=torch.zeros((input_features.shape[0], 1), dtype=torch.long),
        #     encoder_hidden_states=encoder_outputs[0]
        # )
        # logits = self.model.lm_head(decoder_output[0])
        # return logits


torch_model = GenerateWrapper(model, assistant_model).eval()



from torch.export import export


raw_data = input_features.cpu().numpy()

# test_export_and_cuda(raw_data, torch_module)
torch_data = torch.from_numpy(raw_data)

# Give an example argument to torch.export
example_args = (torch_data,)

import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from tvm.relax.frontend.torch import from_exported_program



pytorch_out = torch_model(torch_data).detach().numpy()
print(pytorch_out)