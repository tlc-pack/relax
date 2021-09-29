from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import relax as rx
from tvm import tir, relay

@rx.script
class Mod:
    def foo(x: Tensor[_, "float32"]) -> Shape:
        relax.match_shape(x.shape, (n, m))
        return (n, m)

mod = Mod
print(rx.parser.astext(mod))

new_mod = rx.transform.shape_lower(mod)
print(rx.parser.astext(new_mod))

# @rx.script
# def foo(x: Tensor[_, "float32"]) -> Tensor:
#     x0 = rx.match_shape((n, m), x)
#     y = identity(x0)
#     return y
#     # return (n*2, m*3)


# def test_shape_lower():
#     f = rx_func(foo)
#     match_sh = f.body.blocks[0].bindings[0]
#     print(match_sh)
# 

# def test_shape_lower():
#     shape_anno = [54, 96]
#     type_anno = rx.DynTensorType(2, "float32")
#     x = rx.Var("x", shape_anno, type_anno)
#     m = tir.Var("m", dtype="int32")
#     n = tir.Var("n", dtype="int32")
#     ib = rx.IRBuilder()
#     with ib.function(x):
#         with ib.dataflow() as df:
#             b0 = rx.MatchShape([m, n], x.shape)
#             ib.emit_matchshape(b0)
#             lv0 = rx.call_dps([m, n], rx.extern("test.op.identity"), [x])
#             gv0 = ib.emit_output(lv0)
#         ib.emit_output(gv0)
#     expr = ib.get()
#     print(dir(expr))
#     print(expr.body.blocks[0].bindings[0])

# test_shape_lower()
