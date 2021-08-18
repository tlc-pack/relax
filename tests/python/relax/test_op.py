import tvm
from tvm import tir
from tvm import relax as rx
from tvm.script import ty
from tvm.ir import TensorType
import numpy as np

@tvm.register_func("test.op.identity")
def identity_packed(a):
    return tvm.nd.array(a.asnumpy)

@tvm.script.tir
def identity_tir(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [54, 96])
    B = tir.match_buffer(b, [54, 96])

    with tir.block([54, 96], "compute") as [vi, vj]:
        B[vi, vj] = A[vi, vj]


def test_call_dps() -> None:
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v0 = rx.Var("v0", shape_anno, type_anno)
    v1 = rx.call_dps([54, 96], rx.extern("test.op.identity"), [v0])
    v1 = rx.call_dps([54, 96], identity_tir, [v0])


if __name__ == "__main__":
    test_call_dps()
