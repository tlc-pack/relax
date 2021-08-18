import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import TensorType
import numpy as np

@tvm.register_func("test.op.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)

def test_call_dps() -> None:
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v0 = rx.Var("v0", shape_anno, type_anno)
    shape = rx.ShapeExpr([54, 96])
    packed_func = rx.PackedFuncExpr("test.op.add")
    args = rx.Tuple([v0])
    v1 = rx.op.call_dps(shape, packed_func, args)


if __name__ == "__main__":
    test_call_dps()
