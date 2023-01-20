
import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T


def sch_fn(sch: tir.Schedule):
    t_add = sch.get_block("T_add")
    i0, i1 = sch.get_loops(t_add)
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")


if __name__ == "__main__":    
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(var_rxplaceholder: T.handle, T_add: T.Buffer[(1, 512), "int32"]):
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            data_buf = T.match_buffer(var_rxplaceholder, [1, 512], dtype="int32", align=8)
            # body
            # with T.block("root")
            output_buf = T.alloc_buffer([1, 512], dtype="int32", align=8)
            with T.block("exclusive_scan_thrust"):
                T.reads(data_buf[0, 0 : 512])
                T.writes(output_buf[0, 0 : 512])
                T.tvm_call_packed("tvm.contrib.thrust.sum_scan", T.tvm_stack_make_array(data_buf.data, T.tvm_stack_make_shape(1, 512, dtype="handle"), 0, 2, 0, 0, dtype="handle"), T.tvm_stack_make_array(output_buf.data, T.tvm_stack_make_shape(1, 512, dtype="handle"), 0, 2, 0, 0, dtype="handle"), True, dtype="int32")
            for i0, i1 in T.grid(1, 512):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(data_buf[ax0, ax1], output_buf[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = data_buf[ax0, ax1] + output_buf[ax0, ax1]
    
    mod = Module 
    sch = tvm.tir.Schedule(mod)
    sch_fn(sch)
    sch.mod.show()
