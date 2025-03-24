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


# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
    )


# from hlutils.test_export_and_cuda import test_export_and_cuda

# tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
# if show:
#     tvm_mod.show()

# target = tvm.target.Target.from_device(tvm.cuda())

# ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
# dev = tvm.device("cuda", 0)
# vm = relax.VirtualMachine(ex, dev)

# gpu_data = tvm.nd.array(raw_data, dev)
# gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
# gpu_out = vm["main"](gpu_data, *gpu_params)

# pytorch_out = torch_model(torch_data).detach().numpy()
# actual = gpu_out[0].numpy()
# desired = pytorch_out
# np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5) 
# print("Correctness test passed!") 



# TODO check all this? 
# predicted_ids = torch_module(input_features)
# print("shape of predicted_ids", predicted_ids.shape)
# print("type of predicted_ids", type(predicted_ids))
# print("type of predicted_ids[0]", type(predicted_ids[0]))
# print("predicted_ids[0]", predicted_ids[0])
# print("predicted_ids[1]", predicted_ids[1])

# decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
# print(transcription)
