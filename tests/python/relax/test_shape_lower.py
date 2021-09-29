from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import relax as rx
from tvm import tir, relay
from tvm.script import ty

@rx.script
class Mod:
    def foo(x: Tensor[_, "float32"]) -> Shape:
        relax.match_shape(x.shape, (n, m))
        return (n*2, m*3)

print("Before shape lowering:\n")
mod = Mod()
print(rx.parser.astext(mod))
print("\n")

print("After shape lowering:\n")
new_mod = rx.transform.shape_lower(mod)
print(rx.parser.astext(new_mod))

# @tvm.script.tir
# def shape_func(heap: ty.handle) -> None:
#     H = tir.match_buffer(heap, (5, ), "int64")
#     H[2] = H[0] * tir.int64(2)
#     H[3] = H[1] * tir.int64(3)
# 
# print(shape_func)


# @rx.script
# def foo(x: Tensor[_, "float32"]) -> Tensor:
#     x0 = rx.match_shape((n, m), x)
#     y = identity(x0)
#     return y
#     # return (n*2, m*3)
