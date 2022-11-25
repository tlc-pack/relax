from __future__ import annotations

import math

import numpy as np
import run_cutlass_tuning
import tvm
from scipy.special import erf
from tvm import relax
from tvm.script import relax as R

PKG_FILE = "/tmp/vtx_mm_demo.so"
TARGET = tvm.target.Target("nvidia/nvidia-t4")
M, N, K = 512, 2304, 768

def _print_lowered_mod(mod):
    print(mod["vtx_mm_0"].attrs["c_source"])
    print(mod["vtx_mm_0"].attrs["c_source_fmt"])
    mod.show()


def _relax_build(mod):
    with tvm.transform.PassContext():
        mod = relax.transform.LowerVtxMM()(mod)
        _print_lowered_mod(mod)
        executable = relax.vm.build(mod, target=TARGET)
        executable.mod.export_library(PKG_FILE, cc="nvcc")


def _load_vm():
    executable = tvm.runtime.load_module(PKG_FILE)
    return relax.VirtualMachine(executable, tvm.cuda())


def test_dense():
    # M, N, K = 512, 2304, 768

    @tvm.script.ir_module
    class DenseModule:
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

    mod = DenseModule
    _relax_build(mod)
    vm = _load_vm()

    print("Testing...")
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

    print("Profiling...")
    evaluator = vm.time_evaluator(
        func_name="main",
        dev=tvm.cuda(),
        repeat=10,
        number=10,
        min_repeat_ms=100,
    )
    result = evaluator(a, b)
    print(result)


def test_dense_bias_gelu():
    # M, N, K = 512, 2304, 768

    @tvm.script.ir_module
    class GeluModule:
        @R.function
        def main(
            A: R.Tensor((1, M, K), "float32"),
            B: R.Tensor((1, N, K), "float32"),
            bias: R.Tensor([N], "float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((1, M, N), "float32") = R.vtx_mm(
                    A,
                    B,
                    bias,
                    transpose_a=False,
                    transpose_b=False,
                    epilogue_pattern="cutlass.dense_bias_gelu_fp32",
                )
                R.output(gv)
            return gv

    mod = GeluModule
    _relax_build(mod)
    vm = _load_vm()
    
    print("Testing...")

    def gelu(x):
        return x * 0.5 * (1 + erf(x / math.sqrt(2)))

    a_np = np.random.normal(size=(1, M, K)).astype("float32")
    b_np = np.random.normal(size=(1, N, K)).astype("float32")
    bias_np = np.random.normal(size=(N, )).astype("float32")
    c_np = a_np.reshape(M, K) @ b_np.reshape(N, K).T
    c_np = gelu(c_np + bias_np.reshape(1, N))
    c_np = c_np.reshape(1, M, N)

    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    bias = tvm.nd.array(bias_np, device=tvm.cuda())
    c = vm["main"](a, b, bias)
    c = c.numpy()
    np.testing.assert_allclose(c, c_np, rtol=1e-3, atol=1e-3)
    print("Passed")
    
    print("Profiling...")
    evaluator = vm.time_evaluator(
        func_name="main",
        dev=tvm.cuda(),
        repeat=10,
        number=10,
        min_repeat_ms=100,
    )
    result = evaluator(a, b, bias)
    print(result)


if __name__ == "__main__":
    test_dense()
    test_dense_bias_gelu()
