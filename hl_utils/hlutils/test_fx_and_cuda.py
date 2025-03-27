import sys
sys.path.append('/ssd1/htalendr/tvm/python')
from tvm import relax
import numpy as np
import tvm
from tvm import relax
import torch
from torch import fx
import tvm
import tvm.testing
from tvm.relax.frontend.torch import from_fx

def test_fx_and_cuda(raw_data, torch_model, input_info, show=False, debug=False):
    torch_data = torch.from_numpy(raw_data)

    # Give an example argument to torch.export
    example_args = (torch_data,)

    # Convert the model to IRModule
    # TODO what does , unwrap_unit_return_tuple=True do? should we include?
    with torch.no_grad():
        if debug:
            print("Exporting model...")
        graph_module = fx.symbolic_trace(torch_model)
        if debug:
            print("Converting model to IRModule...")
        mod_from_torch = from_fx(graph_module, input_info)

    if debug:
        print("Detaching parameters...")
    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
    if show:
        tvm_mod.show()

    target = tvm.target.Target.from_device(tvm.cuda())

    if debug:
        print("Defining VM...")
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)

    if debug:
        print("Running VM...")
    gpu_data = tvm.nd.array(raw_data, dev)
    # gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data)

    if debug:
        print("Running PyTorch model...")
    pytorch_out = torch_model(torch_data)

    if debug:
        print("Comparing outputs...")
    if isinstance(pytorch_out, tuple):
        for i in range(len(pytorch_out)):
            actual = gpu_out[i].numpy()
            desired = pytorch_out[i].detach().numpy()
            np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)
    else:
        actual = gpu_out.numpy()
        desired = pytorch_out.detach().numpy()
        np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)

    print("Correctness test passed!") 
