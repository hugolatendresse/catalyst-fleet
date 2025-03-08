import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program


def test_export_and_cuda(raw_data, torch_model, show=False):
    torch_data = torch.from_numpy(raw_data)

    # Give an example argument to torch.export
    example_args = (torch_data,)

    # Convert the model to IRModule
    # TODO what does , unwrap_unit_return_tuple=True do? should we include?
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod_from_torch = from_exported_program(
            exported_program, keep_params_as_input=True#, unwrap_unit_return_tuple=True
        )

    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
    if show:
        tvm_mod.show()

    target = tvm.target.Target.from_device(tvm.cuda())

    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_model(torch_data).detach().numpy()
    actual = gpu_out[0].numpy()
    desired = pytorch_out
    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5) 
    print("Correctness test passed!") 

