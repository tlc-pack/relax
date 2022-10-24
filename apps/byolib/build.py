from __future__ import annotations
import torch

import numpy as np
import tvm
from tvm.script import tir as T, relax as R
from tvm import relax
import tvm.testing


@tvm.script.ir_module
class TestModule:
    # Input IRModule.
    @R.function
    def main(c0: Tensor((5,), "float64")):
        lv0 = relax.call_tir("libtorch_at_abs_out", (c0), (5,), dtype="float64")
        lv1 = relax.call_tir("libtorch_at_tanh_out", (lv0), (5,), dtype="float64")
        return lv1


def create_source_module():
    missing_ops = set(["abs", "tanh"])
    symbols = [f"libtorch_at_{op}_out" for op in missing_ops]
    import toy_parser

    toy_parser.parse_and_gen(missing_ops)
    code = open("byo_libs.cc", "r").read()
    fmt = "cc"

    return tvm.get_global_func("runtime.CSourceModuleCreate")(code, fmt, symbols, [])


def main():
    ex = relax.vm.build(TestModule, target="llvm")

    # create source module and import it as part of ex.mod
    add_on_lib = create_source_module()
    ex.mod.import_module(add_on_lib)

    # NOTE: May need to add extra include and linking options
    # refer to https://github.com/tlc-pack/relax/blob/relax/python/tvm/runtime/module.py#L541
    ex.mod.export_library(
        "packaged.so",
        cc="g++",
        options=[
            "-std=c++17",
            "-I/home/spark/libtorch/include",
            "-L/home/spark/libtorch/lib",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_global_deps",
        ],
    )

    ex = tvm.runtime.load_module("packaged.so")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # verify
    a = np.array([-1.0, -2.0, 3.0, 4.0, -5.0], dtype=np.double)
    tvm_a = tvm.nd.array(a)
    tvm_out = vm["main"](tvm_a)

    torch_a = torch.Tensor(a)
    torch_out = torch.tanh(torch.abs(torch_a))

    tvm.testing.assert_allclose(
        tvm_out.numpy(), torch_out.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


main()
