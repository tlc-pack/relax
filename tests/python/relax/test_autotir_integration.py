import tvm
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
from tvm.relax import meta_schedule as ms
from tvm.tir import Schedule
from tvm.ir.module import IRModule
from tvm.target.target import Target
import tempfile
from typing import List
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord

# Test case with dynamic shape.
# Tuning with dynamic shape is not supported yet.
"""
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
            lv0 = R.call_tir((m,k), tir_matmul, (x,w))
            lv1 = R.call_tir((m,k), tir_relu, (lv0))
            relax.output(lv1)
        return lv1
"""


class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def has_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return True
        return False

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = Workload(mod)
        self.workload_reg.append(workload)
        return workload

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload == workload,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


def test_class_irmodule():
    src = """
class InputModule:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        m = T.var("int32")
        n = T.var("int32")
        k = T.var("int32")
        A = T.match_buffer(x, (16,16))
        B = T.match_buffer(y, (16,16))
        C = T.match_buffer(z, (16,16))

        for (i0, j0, k0) in T.grid(16,16,16):
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
        A = T.match_buffer(x, (16,16))
        B = T.match_buffer(y, (16,16))
        for (i,j) in T.grid(16,16):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x:Tensor[(16,16), "float32"], w:Tensor[(16,16), "float32"]) -> Tensor:
        with R.dataflow():
            sh = relax.call_packed("vm.builtin.shape_of", x)
            x0 = relax.match_shape(sh, (16, 16))
            sh1 = relax.call_packed("vm.builtin.shape_of", w)
            x1 = relax.match_shape(sh1, (16, 16))
            lv0 = R.call_tir((16,16), tir_matmul, (x,w))
            lv1 = R.call_tir((16,16), tir_relu, (lv0))
            relax.output(lv1)
        return lv1
"""
    mod = tvm.script.relax.parser.from_source(src)
    assert isinstance(mod, tvm.IRModule)

    target = Target("llvm --num-cores=16")
    target_host = Target("llvm")

    # Question: Why don't we add target field to ExtractedTask?
    # observer mode (task extraction):
    database = DummyDatabase()
    tasks = ms.integration.extract_task_from_relax(mod, target=target)
    for task in tasks:
        print(f"Extracted task: {task.task_name}, {task.mod}")
        with tempfile.TemporaryDirectory() as work_dir:
            sch: Schedule = tune_tir(
                mod=task.mod,
                target=target,
                config=ReplayTraceConfig(
                    num_trials_per_iter=32,
                    num_trials_total=32,
                ),
                work_dir=work_dir,
                database=database,
            )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)

    with ms.integration.ApplyHistoryBest(database):
        with tvm.transform.PassContext(opt_level=3):
            relax.vm.build(mod, target)


if __name__ == "__main__":
    test_class_irmodule()
