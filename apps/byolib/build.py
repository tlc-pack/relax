from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tir as T, relax as R
from tvm import relax


@tvm.script.ir_module
class TestModule:
    # Input IRModule.
    @R.function
    def main(c0: Tensor((32, 32), "float32"), c1: Tensor((32, 32), "float32")):
        lv0 = relax.call_tir("ext_AddCPU", (c0, c1), (32, 32), dtype="float32")
        return lv0


def create_source_module():
    code = open("byo_libs.cc", "r").read()
    fmt = "cc"
    func_names = ["ext_AddCPU"]
    return tvm.get_global_func("runtime.CSourceModuleCreate")(code, fmt, func_names, [])


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
            "-I/home/yuchenj/libtorch/include",
            "-L/home/yuchenj/libtorch/lib",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_global_deps",
            # "-I/home/yuchenj/downloads/libtorch/include",
            # "/home/yuchenj/downloads/libtorch/lib/libtorch.a",
            # "/home/yuchenj/downloads/libtorch/lib/libc10d.a",
            # "-L/home/yuchenj/downloads/libtorch/lib/",
        ],
    )

    ex = tvm.runtime.load_module("packaged.so")

    vm = relax.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"))
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"))

    nd_res = vm["main"](a, b)
    print(nd_res.numpy())


main()
