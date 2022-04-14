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
""" test tuning pass in relax side"""
from __future__ import annotations
import tvm
from tvm import ir
from tvm.script import tir as T, relax as R
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
import numpy as np
from typing import List
import copy
from tvm.relay.analysis import post_order_visit
from tvm.relay.expr import Call
from tvm.relay.expr_functor import ExprMutator

from tvm.ir.transform import Pass, TuningPass, Trace, Choice, Instruction
from tvm.meta_schedule.database import JSONDatabase
import os, shutil

CONFIG = {"target": "llvm", "target_host": "llvm", "device_id": 0}


@tvm.script.ir_module
class MockModule:
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
            R.output(lv1)
        return lv1


@ir.transform.module_pass(opt_level=0)
class HeuristicPass(Pass):
    def __init__(self, required: List[Pass] = []):
        super().__init__("HeuristicPass", kind=0, required=required)

    # TODO
    # def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
    #    return relay.transform.FoldConstant()(mod)


@ir.transform.module_pass(opt_level=0)
class MockTuningPass(TuningPass):
    def __init__(self, eval_passes=[], required=[], database=None):
        super().__init__(
            "MockTuningPass",
            eval_passes=eval_passes,
            required=required,
            database=database,
        )

    def tune(self, trace, ctx):
        def noapply(mod):
            return mod

        # Create mock choices for testing
        choices = {"c1": Choice(noapply), "c2": Choice(noapply), "c3": Choice(noapply)}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Instruction("SampleMockChoices", choices)

        candidates = knob.generate_candidates(trace)

        candidates = self.consider_eval_passes(candidates, ctx)
        self.evaluate(ctx, candidates)
        best_trace = self.select_best_candidate(candidates)
        return best_trace

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        best_trace = self.tune(Trace(mod), ctx)
        return best_trace.out_mod


# TODO: support ir.Sequential
def apply_sequential(seq, mod):
    cnt = 0
    for p in seq:
        mod = p(mod)
        cnt += p.num_evals
    return mod, cnt


def test_tuning_pass(
    mod=MockModule,
    config=CONFIG,
):
    custom_pass = MockTuningPass()
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3


# Sequential applies each pass one-by-one
# In this case, search space grows in additive manner
def test_sequential(
    mod=MockModule,
    config=CONFIG,
):
    seq = [MockTuningPass(eval_passes=[]), MockTuningPass(eval_passes=[])]
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod, cnt = apply_sequential(seq, mod)

    assert TuningPass.total_num_evals == 3 + 3


# Joint-optimization expands each candidate with its eval_passes
# In this case, search space grows in combinatorial manner
def test_joint_optimization(
    mod=MockModule,
    config=CONFIG,
):
    custom_pass = MockTuningPass(eval_passes=[MockTuningPass(eval_passes=[])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 3

    # Heurstic pass does not affect the search space

    custom_pass = MockTuningPass(eval_passes=[MockTuningPass(eval_passes=[HeuristicPass()])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 3

    custom_pass = MockTuningPass(eval_passes=[MockTuningPass(eval_passes=[MockTuningPass()])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 3 * 3

    custom_pass = MockTuningPass(eval_passes=[MockTuningPass(), MockTuningPass()])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * (3 + 3)

    custom_pass = MockTuningPass(
        eval_passes=[MockTuningPass(eval_passes=[MockTuningPass()]), MockTuningPass()]
    )
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)
    assert TuningPass.total_num_evals == 3 * (3 * 3 + 3)

    custom_pass = MockTuningPass(
        eval_passes=[
            MockTuningPass(
                eval_passes=[MockTuningPass(eval_passes=[MockTuningPass()]), MockTuningPass()]
            )
        ]
    )
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)
    assert TuningPass.total_num_evals == 3 * 3 * (3 * 3 + 3)


def test_database(mod=MockModule, config=CONFIG, remove_after=False):
    def _create_json_database(tmpdir: str) -> JSONDatabase:
        path_workload = os.path.join(tmpdir, "workloads.json")
        path_tuning_record = os.path.join(tmpdir, "tuning_records.json")
        return JSONDatabase(path_workload, path_tuning_record)

    path = "./tmp"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)

    custom_pass = MockTuningPass(database=_create_json_database(path))
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        new_mod = custom_pass(mod)

    if remove_after:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


if __name__ == "__main__":
    test_tuning_pass()
    test_sequential()
    test_joint_optimization()
    test_database()
