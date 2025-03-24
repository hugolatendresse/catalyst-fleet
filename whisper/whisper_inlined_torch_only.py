from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
# encoder = model.get_encoder()

assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

class DecoderWrapper(torch.nn.Module):
    def __init__(self, model, assistant_model):
        super().__init__()
        self.model = model
        self.assistant_model = assistant_model
        
    def forward(self, input_features):
        encoder_outputs = self.model.get_encoder()(input_features)
        decoder_output = self.model.get_decoder()(
            input_ids=torch.zeros((input_features.shape[0], 1), dtype=torch.long), # TODO don't use zeros
            encoder_hidden_states=encoder_outputs[0]
        )
        return decoder_output[0]


torch_model = DecoderWrapper(model, assistant_model).eval()



raw_data = input_features.cpu().numpy()

# test_export_and_cuda(raw_data, torch_module)
torch_data = torch.from_numpy(raw_data)

out = torch_model(torch_data)
print(out)
print("Done!")