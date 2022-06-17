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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tempfile
import numpy as np
from typing import List


import tvm
from tvm import ir, tir
from tvm.ir import transform
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.tir import PrimFunc
from tvm.script import tir as T, relax as R
from tvm import relax
from tvm.relax.expr import Expr, DataflowBlock, Function
import tvm.meta_schedule as ms
from tvm.relax.transform.tuning_api import (
    Choice,
    Knob,
    Trace,
    default_generate_candidate,
)


@tvm.register_func("testing.meta_schedule_tuning_module")
def meta_schedule_tuning_module(mod, target_str):
    target = ms.default_config.target(target_str)
    config = ms.TuneConfig(
        strategy="evolutionary",
        num_trials_per_iter=2,
        max_trials_per_task=4,
        max_trials_global=4,
    )
    database = ms.database.MemoryDatabase()
    with tempfile.TemporaryDirectory() as work_dir:
        extracted_tasks = ms.relax_integration.extract_task_from_relax(mod, target)
        database = ms.tune.tune_extracted_tasks(
            extracted_tasks, config, work_dir, database=database
        )

        return relax.transform.MetaScheduleApplyHistoryBest(database, target)(mod)


@tvm.register_func("testing.meta_schedule_tuning_primfunc")
def meta_schedule_tuning_primfunc(mod, target_str):
    target = ms.default_config.target(target_str)
    config = ms.TuneConfig(
        strategy="evolutionary",
        num_trials_per_iter=2,
        max_trials_per_task=4,
        max_trials_global=4,
    )
    database = ms.database.MemoryDatabase()

    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune.tune_tir(mod, target, config, work_dir, database=database)
        return relax.transform.MetaScheduleApplyHistoryBest(database, target)(mod)


def test_metaschedule_tuning():
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
        def main(x: Tensor((32, 32), "float32"), w: Tensor((32, 32), "float32")) -> Tensor:
            with R.dataflow():
                lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
                lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
                relax.output(lv1)
            return lv1

    mod = InputModule
    assert isinstance(mod, IRModule)
    target_str = "llvm --num-cores=16"

    # One naive implementation of MetaSchedule with module tuning pass.
    # It takes the IRModule, extract tasks from the IRModule, tune each task and apply the best settings.
    @ir.transform.module_pass(opt_level=0, traceable=True)
    def MockMetaSchedTuningPass1(mod: IRModule, ctx: PassContext) -> IRModule:
        trace = ctx.pop_trace()

        # We can create a choice with tuning-based transformation as well.
        choices = [Choice("testing.meta_schedule_tuning_module", [target_str])]
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockMetaSched", choices)

        candidates = default_generate_candidate([knob], trace)
        assert len(candidates) == 1
        best_trace = candidates[0]
        ctx.push_trace(best_trace)
        return best_trace.out_mod

    seq = transform.Sequential([MockMetaSchedTuningPass1])
    with transform.PassContext(trace=Trace(mod)):
        _ = seq(mod)
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 1

    # Another naive implementation of MetaSchedule with prim func tuning pass.
    # It takes each PrimFuncs in IRModule, tune it and apply its best settings.
    @tir.transform.prim_func_pass(opt_level=0, traceable=True)
    def MockMetaSchedTuningPass2(func: PrimFunc, mod: IRModule, ctx: PassContext) -> IRModule:
        trace = ctx.pop_trace()

        # Setup Meta Schedule tuning
        new_mod = ms.default_config.mod(func)
        trace = Trace(new_mod)

        # We can create a choice with tuning-based transformation as well.
        choices = [Choice("testing.meta_schedule_tuning_primfunc", [target_str])]
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockMetaSched", choices)

        candidates = default_generate_candidate([knob], trace)
        assert len(candidates) == 1
        best_trace = candidates[0]
        ctx.push_trace(best_trace)
        gvars = best_trace.out_mod.get_global_vars()
        assert len(gvars) == 1
        return best_trace.out_mod[gvars[0]]

    seq = transform.Sequential([MockMetaSchedTuningPass2])
    with transform.PassContext(trace=Trace(mod)):
        _ = seq(mod)
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 1


if __name__ == "__main__":
    test_metaschedule_tuning()
    # pytest.main([__file__])
