import tvm
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
from tvm import meta_schedule as ms


def test_class_irmodule():
    src = """
class InputModule:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        m = T.var("int32")
        n = T.var("int32")
        k = T.var("int32")
        A = T.match_buffer(x, (m,n))
        B = T.match_buffer(y, (n,k))
        C = T.match_buffer(z, (m,k))

        for (i0, j0, k0) in T.grid(m,n,k):
            with T.block():
                i,j,k = T.axis.remap("SSR", [i0,j0,k0])
                with T.init():
                    C[i,j] = 0.0
                C[i,j] += A[i,k] * B[j,k]

    @T.prim_func
    def tir_relu(x:T.handle, y:T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        m = T.var("int32")
        n = T.var("int32")
        A = T.match_buffer(x, (m,n))
        B = T.match_buffer(y, (m,n))
        for (i,j) in T.grid(m,n):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x:Tensor[(m,n), "float32"], w:Tensor[(n,k), "float32"]) -> Tensor:
        with R.dataflow():
            sh = relax.call_packed("vm.builtin.shape_of", x)
            x0 = relax.match_shape(sh, (m, n))
            sh1 = relax.call_packed("vm.builtin.shape_of", w)
            x1 = relax.match_shape(sh1, (n, k))
            lv0 = R.call_dps((m,k), tir_matmul, (x,w))
            lv1 = R.call_dps((m,k), tir_relu, (lv0))
            relax.output(lv1)
        return lv1
"""
    mod = tvm.script.relax.parser.from_source(src)
    assert isinstance(mod, tvm.IRModule)

    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    # observer mode (task extraction):
    tasks = ms.integration.extract_task(mod, target=target)
    for task in tasks:
        print(f"Extracted task: {task.task_name}, {task.mod}")

    # @sunggg
    # [TODO] injective(Apply) mode: IRModule -> IRModule. (transformed)


if __name__ == "__main__":
    test_class_irmodule()
