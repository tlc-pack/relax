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

import tempfile
import time

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax, transform
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.target.target import Target

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
    def main(x:R.Tensor((m,n), "float32"), w:R.Tensor((n,k), "float32")) -> R.Tensor:
        with R.dataflow():
            sh = R.call_packed("vm.builtin.shape_of", x)
            x0 = R.match_cast(sh, R.Tensor((m, n), "float32"))
            sh1 = R.call_packed("vm.builtin.shape_of", w)
            x1 = R.match_cast(sh1, R.Tensor((n, k), "float32"))
            lv0 = R.call_tir(tir_matmul, (x, w), (m, k), dtype="float32")
            lv1 = R.call_tir(tir_relu, (lv0), (m, k), dtype="float32)
            R.output(lv1)
        return lv1
"""


@pytest.mark.parametrize("dev", ["cpu"])
def test_autotir(dev: str):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
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
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
                lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
                R.output(lv1)
            return lv1

    mod = InputModule
    assert isinstance(mod, IRModule)

    if dev == "cpu":
        target = Target("llvm --num-cores=16")
        dev = tvm.cpu()
    elif dev == "cuda":
        target = Target("nvidia/nvidia-t4")
        dev = tvm.cuda()

    database = ms.database.MemoryDatabase()

    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.relax_integration.tune_relax(
            mod=mod,
            target=target,
            params=None,
            num_trials_per_iter=2,
            max_trials_per_task=4,
            max_trials_global=4,
            work_dir=work_dir,
            database=database,
        )
        relax_ex = ms.relax_integration.compile_relax(
            db,
            mod=mod,
            target=target,
            params=None,
        )

    if dev == "cpu":
        with transform.PassContext(opt_level=3):
            ex0 = relax.vm.build(mod, target)
            vm0 = relax.VirtualMachine(ex0, dev)

        # Measure the performance w/o tuning log
        tic = time.time()
        vm0["main"](data, weight)
        toc = time.time()
        e0 = toc - tic
        print(f"w/o tuning: {e0}")

    vm1 = relax.VirtualMachine(relax_ex, dev)

    data = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)
    weight = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)

    # Measure the performance w/ tuning log
    tic = time.time()
    vm1["main"](data, weight)
    toc = time.time()
    e1 = toc - tic
    print(f"w/  tuning: {e1}")


@tvm.testing.requires_gpu
def test_autotir_gpu():
    test_autotir("cuda")


def test_meta_schedule_extract_tasks():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add1(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
            for i, j in T.grid(128, 128):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 1.0

        @T.prim_func
        def add2(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
            for i, j in T.grid(128, 128):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 2.0

        # It is intentional that `add3` equals `add1`, in order to test the deduplication
        # correctness.
        @T.prim_func
        def add3(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
            for i, j in T.grid(128, 128):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 1.0

        @T.prim_func
        def multiply1(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
            for i, j in T.grid(128, 128):
                with T.block("multiply"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

        @R.function
        def main(x: R.Tensor((128, 128), "float32")) -> R.Tensor(dtype="float32"):
            with R.dataflow():
                lv0 = R.call_tir(add1, (x,), (128, 128), dtype="float32")
                lv1 = R.call_tir(multiply1, (lv0,), (128, 128), dtype="float32")
                lv2 = R.call_tir(add2, (lv1,), (128, 128), dtype="float32")
                lv3 = R.call_tir(multiply1, (lv2,), (128, 128), dtype="float32")
                lv4 = R.call_tir(add3, (lv3,), (128, 128), dtype="float32")
                gv = R.call_tir(add1, (lv4,), (128, 128), dtype="float32")
                R.output(gv)
            return gv

    tasks = ms.relax_integration.extract_tasks(Module, Target("llvm --num-cores=16"))
    expected_weights = {"add1": 3, "add2": 1, "multiply1": 2}
    assert len(tasks) == len(expected_weights)
    for task in tasks:
        assert task.task_name in expected_weights
        assert expected_weights[task.task_name] == task.weight


if __name__ == "__main__":
    pytest.main([__file__])
