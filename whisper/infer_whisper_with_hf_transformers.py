from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
import torch
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# print("model is a", type(model))
# predicted_ids = model.generate(input_features, assistant_model=assistant_model)

class GenerateWrapper(torch.nn.Module):
    def __init__(self, model, assistant_model):
        super().__init__()
        self.model = model
        self.assistant_model = assistant_model

    def forward(self, input_features):
        return self.model.generate(input_features, assistant_model=self.assistant_model)

# TODO pretty sure the eval() below does nothing since I don't think it affects the model attribute? OR DOES IT? need to check
torch_module = GenerateWrapper(model, assistant_model).eval()

predicted_ids = torch_module(input_features)

print("shape of predicted_ids", predicted_ids.shape)
print("type of predicted_ids", type(predicted_ids))
print("type of predicted_ids[0]", type(predicted_ids[0]))
print("predicted_ids[0]", predicted_ids[0])
print("predicted_ids[1]", predicted_ids[1])

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
