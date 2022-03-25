# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations
import tvm
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
from tvm.tir import Schedule
from tvm.ir.module import IRModule
from tvm.target.target import Target
import tempfile
from typing import List
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.meta_schedule.integration import extract_task_from_relax
from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.search_strategy import EvolutionarySearchConfig
from tvm import transform
import time
import pytest


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
            lv0 = R.call_tir(tir_matmul, (x, w), (m, k), dtype="float32")
            lv1 = R.call_tir(tir_relu, (lv0), (m, k), dtype="float32)
            relax.output(lv1)
        return lv1
"""


@derived_object
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


@pytest.mark.parametrize("dev", ["cpu"])
def test_autotir(dev: str):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (32, 32))
            B = T.match_buffer(y, (32, 32))
            C = T.match_buffer(z, (32, 32))

            for (i0, j0, k0) in T.grid(32, 32, 32):
                with T.block():
                    i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                    with T.init():
                        C[i, j] = 0.0
                    C[i, j] += A[i, k] * B[j, k]

        @T.prim_func
        def tir_relu(x: T.handle, y: T.handle):
            T.func_attr({"global_symbol": "tir_relu"})
            m = T.var("int32")
            n = T.var("int32")
            A = T.match_buffer(x, (32, 32))
            B = T.match_buffer(y, (32, 32))
            for (i, j) in T.grid(32, 32):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = T.max(A[vi, vj], 0.0)

        @R.function
        def main(x: Tensor[(32, 32), "float32"], w: Tensor[(32, 32), "float32"]) -> Tensor:
            with R.dataflow():
                lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
                lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
                relax.output(lv1)
            return lv1

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    if dev == "cpu":
        target = Target("llvm --num-cores=16")
        dev = tvm.cpu()
    else:
        target = Target("nvidia/geforce-rtx-3070")
        dev = tvm.cuda()

    database = DummyDatabase()
    tasks = extract_task_from_relax(mod, target=target)
    for task in tasks:
        print(f"Extracted task: {task.task_name}, {task.target}")
        with tempfile.TemporaryDirectory() as work_dir:
            sch = tune_tir(
                mod=task.mod,
                target=target,
                config=EvolutionarySearchConfig(
                    num_trials_per_iter=32,
                    num_trials_total=32,
                ),
                work_dir=work_dir,
                database=database,
            )

    with transform.PassContext(opt_level=3):
        ex0 = relax.vm.build(mod, target)

    with transform.PassContext(opt_level=3):
        mod = relax.transform.MetaScheduleApplyHistoryBest(database, target)(mod)
        ex1 = relax.vm.build(mod, target)

    vm0 = relax.VirtualMachine(ex0, dev)
    vm1 = relax.VirtualMachine(ex1, dev)
    data = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)
    weight = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)

    # Measure the performance w/o tuning log
    tic = time.time()
    vm0["main"](data, weight)
    toc = time.time()
    e0 = toc - tic

    # Measure the performance w/ tuning log
    tic = time.time()
    vm1["main"](data, weight)
    toc = time.time()
    e1 = toc - tic

    print(f"w/o tuning: {e0}")
    print(f"w/  tuning: {e1}")


if __name__ == "__main__":
    pytest.main([__file__])
