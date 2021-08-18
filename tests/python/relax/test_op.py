import tvm
from tvm import tir
from tvm import relax as rx
from tvm.script import ty
from tvm.ir import TensorType
import numpy as np

@tvm.register_func("test.op.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)

@tvm.script.tir
def identity(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [54, 96])
    B = tir.match_buffer(b, [54, 96])

    with tir.block([54, 96], "compute") as [vi, vj]:
        B[vi, vj] = A[vi, vj]


def test_call_dps() -> None:
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v0 = rx.Var("v0", shape_anno, type_anno)
    shape = rx.ShapeExpr([54, 96])
    args = rx.Tuple([v0])

    packed_func = rx.PackedFuncExpr("test.op.add")
    v1 = rx.op.call_dps(shape, packed_func, args)

    tir_func = identity
    v1 = rx.op.call_dps(shape, tir_func, args)


if __name__ == "__main__":
    test_call_dps()
