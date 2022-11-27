# pylint: disable=missing-docstring
import tempfile

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T

# pylint: disable=invalid-name


@tvm.script.ir_module
class Module1:
    @T.prim_func
    def main(
        lv69: T.Buffer[(1, 768), "float32"],
        param_0: T.Buffer[(T.int64(768), T.int64(768)), "float32"],
        param_1: T.Buffer[T.int64(768), "float32"],
        T_concat: T.Buffer[(1, 768), "float32"],
    ):
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        T_matmul_NT = T.alloc_buffer([1, 768], dtype="float32")
        T_add = T.alloc_buffer([1, 768], dtype="float32")
        T_minimum = T.alloc_buffer([1, 768], dtype="float32")
        T_maximum = T.alloc_buffer([1, 768], dtype="float32")
        T_fast_tanh = T.alloc_buffer([1, 768], dtype="float32")
        for i0, i1, i2 in T.grid(1, 768, 768):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(lv69[i, k], param_0[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + lv69[i, k] * param_0[j, k]
        for i0, i1 in T.grid(1, 768):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], param_1[ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + param_1[ax1]
        for i0, i1 in T.grid(1, 768):
            with T.block("T_minimum"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add[ax0, ax1])
                T.writes(T_minimum[ax0, ax1])
                T_minimum[ax0, ax1] = T.min(T.float32(9), T_add[ax0, ax1])
        for i0, i1 in T.grid(1, 768):
            with T.block("T_maximum"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_minimum[ax0, ax1])
                T.writes(T_maximum[ax0, ax1])
                T_maximum[ax0, ax1] = T.max(T.float32(-9), T_minimum[ax0, ax1])
        for i0, i1 in T.grid(1, 768):
            with T.block("T_fast_tanh"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_maximum[ax0, ax1])
                T.writes(T_fast_tanh[ax0, ax1])
                # fmt: off
                T_fast_tanh[ax0, ax1] = T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * T.float32(-2.76076847742355e-16) + T.float32(2.0001879048247699e-13)) + T.float32(-8.60467152213735e-11)) + T.float32(5.1222970903711401e-08)) + T.float32(1.4857223571797901e-05)) + T.float32(0.00063726192887543596)) + T.float32(0.0048935245589178597)) / (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * (T_maximum[ax0, ax1] * T_maximum[ax0, ax1] * T.float32(1.1982583946670199e-06) + T.float32(0.000118534705686654)) + T.float32(0.0022684346324390002)) + T.float32(0.0048935251855438504))
                # fmt: on
        for i0, i1 in T.grid(1, 768):
            with T.block("T_concat"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_fast_tanh[ax0, ax1])
                T.writes(T_concat[ax0, ax1])
                T_concat[ax0, ax1] = T_fast_tanh[ax0, ax1]


@tvm.script.ir_module
class Module2:
    @T.prim_func
    def main(
        lv73: T.Buffer[(1, 768), "float32"],
        param_0: T.Buffer[(T.int64(200), T.int64(768)), "float32"],
        param_1: T.Buffer[T.int64(200), "float32"],
        compute: T.Buffer[(1, 200), "float32"],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "fused_dense1_add3_relu"})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([1, 200], dtype="float32")
        T_add = T.alloc_buffer([1, 200], dtype="float32")
        for i0, i1, i2 in T.grid(1, 200, 768):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(lv73[i, k], param_0[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + lv73[i, k] * param_0[j, k]
        for i0, i1 in T.grid(1, 200):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], param_1[ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + param_1[ax1]
        for i0, i1 in T.grid(1, 200):
            with T.block("compute"):
                i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add[i0_1, i1_1])
                T.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(T_add[i0_1, i1_1], T.float32(0))


@tvm.script.ir_module
class Module3:
    @T.prim_func
    def main(
        lv76: T.Buffer[(1, 200), "float32"],
        param_0: T.Buffer[(T.int64(20), T.int64(200)), "float32"],
        param_1: T.Buffer[T.int64(20), "float32"],
        T_add: T.Buffer[(1, 20), "float32"],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "fused_dense2_add4"})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([1, 20], dtype="float32")
        for i0, i1, i2 in T.grid(1, 20, 200):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(lv76[i, k], param_0[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + lv76[i, k] * param_0[j, k]
        for i0, i1 in T.grid(1, 20):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], param_1[ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + param_1[ax1]


@tvm.script.ir_module
class Module4:
    @T.prim_func
    def main(
        lv76: T.Buffer[(1, 200), "float32"],
        param_0: T.Buffer[(T.int64(20), T.int64(200)), "float32"],
        param_1: T.Buffer[T.int64(20), "float32"],
        compute: T.Buffer[(1, 20), "float32"],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "fused_dense2_add4_relu1"})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([1, 20], dtype="float32")
        T_add = T.alloc_buffer([1, 20], dtype="float32")
        for i0, i1, i2 in T.grid(1, 20, 200):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(lv76[i, k], param_0[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + lv76[i, k] * param_0[j, k]
        for i0, i1 in T.grid(1, 20):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], param_1[ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + param_1[ax1]
        for i0, i1 in T.grid(1, 20):
            with T.block("compute"):
                i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add[i0_1, i1_1])
                T.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(T_add[i0_1, i1_1], T.float32(0))


@tvm.script.ir_module
class Module5:
    @T.prim_func
    def main(
        lv81: T.Buffer[(1, 20), "float32"],
        param_0: T.Buffer[(T.int64(1), T.int64(20)), "float32"],
        param_1: T.Buffer[T.int64(1), "float32"],
        T_add: T.Buffer[(1, 1), "float32"],
    ):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "fused_dense3_add5"})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([1, 1], dtype="float32")
        for i0, i1, i2 in T.grid(1, 1, 20):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(lv81[i, k], param_0[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + lv81[i, k] * param_0[j, k]
        for i0, i1 in T.grid(1, 1):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], param_1[ax1])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + param_1[ax1]


def sch_fn(sch: tir.Schedule):
    gemv = sch.get_block("T_matmul_NT")
    epilogue = sch.cache_write(gemv, 0, "local")
    i0, i1, tx = sch.get_loops(gemv)
    bx = sch.fuse(i0, i1)
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.reverse_compute_at(epilogue, bx)
    while True:
        consumers = sch.get_consumers(epilogue)
        if consumers:
            sch.reverse_compute_inline(consumers[0])
        else:
            break
    # sch.annotate(
    #     sch.get_block("root"),
    #     ann_key="meta_schedule.unroll_explicit",
    #     ann_val=sch.sample_categorical([0, 16, 64, 512, 1024], [0.2, 0.2, 0.2, 0.2, 0.2]),
    # )


def main():
    target = tvm.target.Target("nvidia/nvidia-t4")
    for mod in [Module1, Module2, Module3, Module4, Module5]:
        sch = tir.Schedule(mod)
        sch_fn(sch)
        sch.mod.show()
        tvm.build(sch.mod, target=target)

    # a_np = np.random.uniform(size=(1, 768), low=-1.0, high=1.0).astype("float32")
    # b_np = np.random.uniform(size=(768, 768)).astype("float32")
    # c_np = np.random.uniform(size=(768)).astype("float32")
    # d_np = np.random.uniform(size=(1, 768)).astype("float32")
    #
    # a = tvm.nd.array(a_np, tvm.cuda(0))
    # b = tvm.nd.array(b_np, tvm.cuda(0))
    # c = tvm.nd.array(c_np, tvm.cuda(0))
    # d = tvm.nd.array(d_np, tvm.cuda(0))
    # mod(a, b, c, d)
    # np.testing.assert_allclose(b.numpy(), a_np.sum(), atol=1e-3, rtol=1e-3)
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.tir_integration.tune_tir(
            Module5,
            target=target,
            work_dir=work_dir,
            max_trials_global=500,
            space=sch_fn,
        )
    # sch = db.query_schedule(Module, target, workload_name="main")
    # sch.mod.show()
    # print(sch.trace)


if __name__ == "__main__":
    main()
