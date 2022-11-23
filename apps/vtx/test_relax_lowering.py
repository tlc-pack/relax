from __future__ import annotations

import numpy as np
import run_cutlass_tuning
import tvm
from tvm import relax
from tvm.script import relax as R

PKG_FILE = "/tmp/vtx_mm_demo.so"

M = 512
K = 768
N = 2304


@tvm.script.ir_module
class TestModule:
    @R.function
    def main(
        A: R.Tensor((1, M, K), "float32"),
        B: R.Tensor((1, N, K), "float32"),
    ):
        with R.dataflow():
            gv: R.Tensor((1, M, N), "float32") = R.vtx_mm(
                A,
                B,
                transpose_a=False,
                transpose_b=False,
                epilogue_pattern="",
            )
            R.output(gv)
        return gv


def _print_lowered_mod(mod):
    print(mod["vtx_mm_0"].attrs["c_source"])
    print(mod["vtx_mm_0"].attrs["c_source_fmt"])
    mod.show()


def _load_vm(executable):
    executable.mod.export_library(PKG_FILE, cc="nvcc")
    executable = tvm.runtime.load_module(PKG_FILE)
    return relax.VirtualMachine(executable, tvm.cuda())


def main():
    target = tvm.target.Target("cuda -arch=sm_80 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32 -registers_per_block=65536")
    mod = TestModule
    with tvm.transform.PassContext():
        mod = relax.transform.LowerVtxMM()(mod)
        _print_lowered_mod(mod)
        executable = relax.vm.build(mod, target=target)
    vm = _load_vm(executable)

    a_np = np.random.rand(1, M, K).astype("float32")
    b_np = np.random.rand(1, N, K).astype("float32")
    c_np = np.random.rand(1, M, N).astype("float32")
    c_np = a_np.reshape(M, K) @ b_np.reshape(N, K).T
    c_np = c_np.reshape(1, M, N)

    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    c = vm["main"](a, b)
    c = c.numpy()
    np.testing.assert_allclose(c, c_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    main()
