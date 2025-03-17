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

predicted_ids = GenerateWrapper(model, assistant_model)(input_features)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
