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
    default_consider_eval_passes,
    default_evaluate,
    select_best_candidate,
    get_trace,
)


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def addone(A: T.Buffer[(16, 16), "int32"], B: T.Buffer[(16, 16), "int32"]) -> None:
        T.func_attr(({"global_symbol": "addone"}))
        for i, j in T.grid(16, 16):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.int32(1)

    @R.function
    def main(c0: Tensor((16, 16), "int32")):
        lv0 = relax.call_tir(addone, (c0,), (16, 16), dtype="int32")
        return lv0


# Setup for testing with constant folding.
def setup_test_const_folding():
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def addone(A: T.Buffer[(16, 16), "int32"], B: T.Buffer[(16, 16), "int32"]) -> None:
            T.func_attr(({"global_symbol": "addone"}))
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        # Input IRModule.
        @R.function
        def before(c0: Tensor((16, 16), "int32")):
            lv0 = relax.call_tir(addone, (c0,), (16, 16), dtype="int32")
            return lv0

        # Expected IRModule after transformation.
        @R.function
        def expected(c1: Tensor((16, 16), "int32")):
            lv0 = c1
            return c1

    def gen_mod(mod, name, binding):
        funcs = {}
        binding = {k: tvm.nd.array(v) for k, v in binding.items()}

        for k, v in mod.functions.items():
            if isinstance(v, tvm.relax.Function):
                if k.name_hint == name:
                    # rename to main.
                    gv = tvm.ir.GlobalVar("main")
                    funcs[gv] = tvm.relax.Function(v.params, v.body, v.ret_type).with_attr(
                        "global_symbol", "main"
                    )
            else:
                funcs[k] = v
        mod = tvm.IRModule(funcs)
        return relax.transform.BindParams("main", binding)(mod)

    mod = TestModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup.
    c0_np = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(mod, "before", {"c0": c0_np})
    expected = gen_mod(mod, "expected", {"c1": c1_np})

    return before, expected


# Mock evaluation pass for testing.
# Assigns arbitrary performance number to each candidate.
def mock_evaluate(candidates: List[Trace], target_str: str, ctx: PassContext):
    num_evals = 0
    # Evaluation
    for candidate in candidates:
        # If this candidate is already evaluated, skip the measurement.
        if candidate.perf != -1:
            continue

        num_evals += 1
        # Assign arbitrary performance.
        mock_perf = 100 - (ctx.num_evals + num_evals)
        candidate.set_perf(mock_perf)
    # Update number of evals for testing.
    ctx.inc_num_evals(num_evals)


# Define a choice by using FoldConstant pass.
@tvm.register_func("testing.apply_fold_constant")
def apply_fold_constant(mod):
    return relax.transform.FoldConstant()(mod)


@tvm.register_func("testing.add_global_symbol")
def add_global_symbol(mod, func_name, global_symbol):
    mod[func_name] = mod[func_name].with_attr("global_symbol", global_symbol)
    return mod


@tvm.register_func("testing.check_num_functions")
def check_num_funcs(mod, N):
    # Explicit type specification is necessary.
    # Otherwise, PackedFunc cannot derive the return type correctly.
    # e.g., Check failed: type_code_ == kDLInt (8 vs. 0) : expected int but got Object
    return bool(len(mod.functions) == N)


def test_choice():
    # Test setup.
    (
        before,
        expected,
    ) = setup_test_const_folding()

    # Without any argument, default setting will be used for both transformation and constraint functions.
    # default transformation function will return the original IRModule without any change.
    choice = Choice(
        # - f_transform_key="relax.tuning_api.Choice.f_default_transform"
        # - f_constr_key="relax.tuning_api.Choice.f_default_constr")
    )
    # Load transformation function from the choice and apply it.
    after = choice.apply_transform_func(before)
    tvm.ir.assert_structural_equal(after, before)

    choice = Choice("testing.apply_fold_constant")
    # Load transformation function from the choice and apply it.
    after = choice.apply_transform_func(before)
    tvm.ir.assert_structural_equal(after, expected)

    # Create a choice that tags global symbol onto target function.
    choice = Choice("testing.add_global_symbol", ["addone", "test-symbol"])
    after = choice.apply_transform_func(before)
    assert after["addone"].attrs["global_symbol"] == "test-symbol"
    # The transformation should be applied with Copy-On-Write.
    # So, the original module should be unchanged.
    assert before["addone"].attrs["global_symbol"] == "addone"

    # Test choice with impossible constraint
    choice = Choice(
        f_transform_key="testing.add_global_symbol",
        f_transform_args=["addone", "test-symbol"],
        f_constr_key="testing.check_num_functions",
        f_constr_args=[1000],
    )
    # Since the constraint is not met, it should return the original function
    after = choice.apply_transform_func(before)
    assert after["addone"].attrs["global_symbol"] == "addone"

    # Test choice with the proper constraint
    choice = Choice(
        f_transform_key="testing.add_global_symbol",
        f_transform_args=["addone", "test-symbol"],
        f_constr_key="testing.check_num_functions",
        f_constr_args=[2],
    )
    # Since the constraint is not met, it should return the original function
    after = choice.apply_transform_func(before)
    assert after["addone"].attrs["global_symbol"] == "test-symbol"
    # The original module should be unchanged.
    assert before["addone"].attrs["global_symbol"] == "addone"

    # Test roundtrip.
    # Export as JSON.
    json_obj = choice.as_json()
    # Import JSON.
    new_choice = Choice.from_json(json_obj)
    # Test imported choice
    after = new_choice.apply_transform_func(before)
    assert after["addone"].attrs["global_symbol"] == "test-symbol"
    # The original module should be unchanged.
    assert before["addone"].attrs["global_symbol"] == "addone"


def test_knob():
    # Test setup.
    before, expected = setup_test_const_folding()

    # Users can define a set of choices with list.
    choices = [
        Choice("testing.apply_fold_constant"),
        Choice(),
    ]

    # Define knob.
    knob = Knob("TestKnob", choices)
    # Check the sanity of decision space.
    assert knob.verify(0)
    assert knob.verify(1)
    assert not knob.verify(3)

    # Check the sanity of each decision.
    after_apply = knob.apply(before, 0)
    after_noapply = knob.apply(before, 1)

    tvm.ir.assert_structural_equal(after_apply, expected)
    tvm.ir.assert_structural_equal(after_noapply, before)

    # Users can define a set of choices with dict.
    choices = {
        "apply": Choice("testing.apply_fold_constant"),
        "noapply": Choice(),
        "apply_with_impossible_constr": Choice(
            f_transform_key="testing.apply_fold_constant",
            f_constr_key="testing.check_num_functions",
            f_constr_args=[1000],
        ),
    }
    # Define knob.
    knob = Knob("TestKnob", choices)
    assert knob.verify("apply")
    assert knob.verify("noapply")
    assert knob.verify("apply_with_impossible_constr")
    assert not knob.verify("INVLAID")

    after_apply = knob.apply(before, "apply")
    after_noapply = knob.apply(before, "noapply")
    # Because constr was not satisfied, it will return the original IRModule
    after_apply_with_constr = knob.apply(before, "apply_with_impossible_constr")
    tvm.ir.assert_structural_equal(after_apply, expected)
    tvm.ir.assert_structural_equal(after_noapply, before)
    tvm.ir.assert_structural_equal(after_apply_with_constr, before)

    # Test roundtrip.
    # Export as JSON.
    json_obj = knob.as_json()
    # Import JSON.
    new_knob = Knob.from_json(json_obj)
    assert new_knob.name == knob.name
    # Test imported knob
    assert new_knob.verify("apply")
    assert new_knob.verify("noapply")
    assert new_knob.verify("apply_with_impossible_constr")
    assert not new_knob.verify("INVLAID")

    after_apply = new_knob.apply(before, "apply")
    after_noapply = new_knob.apply(before, "noapply")
    # Because constr was not satisfied, it will return the original IRModule
    after_apply_with_constr = knob.apply(before, "apply_with_impossible_constr")
    tvm.ir.assert_structural_equal(after_apply, expected)
    tvm.ir.assert_structural_equal(after_noapply, before)
    tvm.ir.assert_structural_equal(after_apply_with_constr, before)


def test_trace():
    before, expected = setup_test_const_folding()

    # Define choices and its knob.
    choices = {
        "apply": Choice(
            f_transform_key="testing.apply_fold_constant",
            f_transform_args=[],
            f_constr_key="testing.check_num_functions",
            f_constr_args=[2],
        ),
        "noapply": Choice(),
    }
    knob = Knob("TestKnob", choices)

    # Define a Trace with empty decision (transformation) history.
    trace = Trace(before)
    assert trace.size == 0

    # Define a Trace with single decision (transformation) history.
    trace = Trace(before, [knob], ["noapply"])
    assert trace.size == 1
    tvm.ir.assert_structural_equal(trace.in_mod, before)
    tvm.ir.assert_structural_equal(trace.out_mod, before)

    # Add a new knob and its decision to the trace.
    # It will update the current trace and returns its new output IRModule.
    out: IRModule = trace.add(knob, "noapply")
    assert trace.size == 2
    tvm.ir.assert_structural_equal(trace.in_mod, before)
    tvm.ir.assert_structural_equal(trace.out_mod, before)
    tvm.ir.assert_structural_equal(out, before)
    # Assume we assign arbitrary performance number.
    trace.set_perf(100)
    assert trace.perf == 100

    # Add a new knob and its decision to the trace.
    out: IRModule = trace.add(knob, "apply")
    tvm.ir.assert_structural_equal(trace.in_mod, before)
    tvm.ir.assert_structural_equal(trace.out_mod, expected)
    tvm.ir.assert_structural_equal(out, expected)

    assert trace.size == 3
    # Should be initalized when new knob is applied.
    assert trace.perf == -1

    # Test roundtrip.
    # Export as JSON.
    json_obj = trace.as_json()
    # Import JSON.
    new_trace = Trace.from_json(json_obj)
    tvm.ir.assert_structural_equal(trace.in_mod, new_trace.in_mod)
    assert new_trace.size == 3
    tvm.ir.assert_structural_equal(trace.out_mod, new_trace.out_mod)


def test_trace_wrapper():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    assert isinstance(Trace(mod), Trace)
    assert isinstance(get_trace(mod), Trace)
    assert isinstance(get_trace(mod["main"]), Trace)
    assert isinstance(get_trace(mod["addone"]), Trace)


# Mock tuning pass that determines whether to apply relax.transform.FoldConstant().
# Each pass invocation will generate two candidates for the incoming IRModule.
# In relax pass infra, each pass will define its own way of generating candidates and evaluating them without needing to know how other passes generate its candidate and evaluate them.
# This will significantly alleviate the development process since it is known to be HARD problem to consider the interaction with (potentially hundreds of) other passes.
@ir.transform.module_pass(opt_level=0, traceable=True)
class MockConstFoldingTuningPass(transform.Pass):
    def __init__(
        self,
        f_generate_candidate=None,
        f_evaluate=mock_evaluate,
        eval_passes: List[transform.Pass] = None,
        required: List[transform.Pass] = [],
    ):
        self.f_generate_candidate = (
            f_generate_candidate if f_generate_candidate else default_generate_candidate
        )
        self.f_evaluate = f_evaluate if f_evaluate else default_evaluate
        self.eval_passes = eval_passes
        self.required = required

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        trace = ctx.pop_trace()

        # Create mock choices for testing.
        choices = {"apply": Choice("testing.apply_fold_constant"), "noapply": Choice()}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockTuningKnob", choices)

        candidates = self.f_generate_candidate([knob], trace, self.eval_passes)
        self.f_evaluate(candidates, "llvm", ctx)
        best_trace = select_best_candidate(candidates)

        ctx.push_trace(best_trace)
        return best_trace.out_mod


def test_default_functions():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)

    # Define choice, knob, trace.
    choices = {"apply": Choice("testing.apply_fold_constant"), "noapply": Choice()}
    knob = Knob("TestKnob", choices)
    trace = Trace(mod)

    # Launch a pass pipeline in trace mode.
    with transform.PassContext(trace=trace):
        # Default generation function expands every valid choice.
        candidates = default_generate_candidate([knob], trace)
        assert len(candidates) == 2

        # Default evaluate function uses MetaSchedule builder/runner.
        # Since builder/runner are not provided, local builder/runner will be used.
        default_evaluate(candidates, "llvm --num-cores=16")
        assert PassContext.current().num_evals == 2

        # Because these candidates are already evaluated, num_evals stays the same.
        default_evaluate(candidates, "llvm --num-cores=16")
        assert PassContext.current().num_evals == 2

        # Test with multiple knobs
        candidates = default_generate_candidate([knob, knob], trace)
        assert len(candidates) == 4

    # Launch new pass pipeline in trace mode.
    with transform.PassContext(trace=trace):
        candidates = default_generate_candidate([knob], trace)
        assert len(candidates) == 2
        # Provide tuning pass as an eval pass.
        # Note that MockConstFoldingTuningPass() has its own generation function, evaluation function.
        # Evaluation would be done in a tornament fashion.
        # `default_consider_eval_passes` will convert candidates into the best version by considering eval_passes.
        # For example, if we say candidates = [C1, C2]
        # `default_consider_eval_passes` will return best form of C1 variant (C11 vs C12) and C2 variant (C21 vs C22)
        # that can be generated by eval_passes.
        # Assume C11 > C12, C21 < C22,
        # new_candidates = [C11, C22]
        new_candidates = default_consider_eval_passes(
            candidates, [MockConstFoldingTuningPass(eval_passes=[])]
        )

        # len(candidates) == len(new candidates).
        assert len(new_candidates) == 2
        # To find the best version of each candidate, it would take 4 evals (C11, C12, C21, C22).
        assert PassContext.current().num_evals == 4

    HeuristicPass = relax.transform.FoldConstant
    with transform.PassContext(trace=trace):
        candidates = default_generate_candidate([knob], trace)
        assert len(candidates) == 2
        # Provide heuristic pass as an eval pass.
        new_candidates = default_consider_eval_passes(candidates, [HeuristicPass()])
        # Since heuristic pass has single decision, it won't need any tornament.
        # new_candidates = [C11, C21]
        assert len(new_candidates) == 2
        # We only conduct evaluation when its necessary (e.g., choose better candidate in tuning pass).
        # Heuristic pass won't conduct any evaluation.
        assert PassContext.current().num_evals == 0


# TODO(sunggg): Do we need to serialize pass context as well?
def test_pass_context():
    before, expected = setup_test_const_folding()
    HeuristicPass = relax.transform.FoldConstant
    # FoldConstant implicitly performs TIR passes (prob for constant evaluation).
    # If make_traceable is not provided, the pass infra will make every non-traceable pass traceable by default.
    seq = transform.Sequential([HeuristicPass()])
    with transform.PassContext(
        trace=Trace(before),
    ):
        after = seq(before)
        tvm.ir.assert_structural_equal(after, expected)
        assert PassContext.current().get_trace_stack_size() == 1
        # The exact number of implicit passes might change as TVM develops more passes.
        # As of today, this size returns 57.
        assert PassContext.current().get_current_trace().size > 1

    # We can explicitly specify which pass we want to keep track of.
    with transform.PassContext(trace=Trace(before), make_traceable=["FoldConstant"]):
        after = seq(before)
        tvm.ir.assert_structural_equal(after, expected)
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 1

    # Check the functionality of trace stack.
    with transform.PassContext(trace=Trace(before)):
        assert PassContext.current().get_trace_stack_size() == 1
        PassContext.current().push_trace(Trace(before))
        assert PassContext.current().get_trace_stack_size() == 2
        PassContext.current().pop_trace()
        assert PassContext.current().get_trace_stack_size() == 1
        PassContext.current().pop_trace()
        assert PassContext.current().get_trace_stack_size() == 0


def test_module_pass():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup
    c0 = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    mod = relax.transform.BindParams("main", {"c0": tvm.nd.array(c0)})(mod)
    HeuristicPass = relax.transform.FoldConstant

    # Tuning pass without any eval_pass.
    mock_pass = MockConstFoldingTuningPass(eval_passes=[])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 1

    # Heuristic pass should not affect the number of candidates.
    mock_pass = MockConstFoldingTuningPass(eval_passes=[HeuristicPass()])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 2

    # Joint-optimization will increase the search space in the combinatorial way
    mock_pass = MockConstFoldingTuningPass(eval_passes=[MockConstFoldingTuningPass(eval_passes=[])])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 2

    # Joint-optimization can be nested.
    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            MockConstFoldingTuningPass(eval_passes=[MockConstFoldingTuningPass(eval_passes=[])])
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2 * 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 3

    # Tuning pass and heuritic passes can be used together.
    # Note that heuristic pass won't increate the search space (num_evals).
    # It only increases the length of the trace.
    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            HeuristicPass(),
            MockConstFoldingTuningPass(
                eval_passes=[
                    MockConstFoldingTuningPass(eval_passes=[HeuristicPass(), HeuristicPass()])
                ]
            ),
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2 * 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 6

    # Users can mix-use sequential application and joint-application.
    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * (2 + 2 + 2)
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 4


def test_sequential():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup.
    c0 = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    mod = relax.transform.BindParams("main", {"c0": tvm.nd.array(c0)})(mod)
    HeuristicPass = relax.transform.FoldConstant

    # Sequential with a single tuning pass should behave same with a single pass.
    seq = transform.Sequential([MockConstFoldingTuningPass(eval_passes=[])])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 1

    # Sequential pass should increase search space (num_evals) in additive manner.
    seq = transform.Sequential(
        [
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2 + 2 + 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 3

    # Heuristic pass will not increase the search space. Just increase trace length.
    seq = transform.Sequential(
        [
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2 + 2 + 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 5

    # Users can mix-use sequential application and joint-application.
    seq = transform.Sequential(
        [
            HeuristicPass(),
            MockConstFoldingTuningPass(
                eval_passes=[
                    MockConstFoldingTuningPass(
                        eval_passes=[
                            MockConstFoldingTuningPass(
                                eval_passes=[
                                    HeuristicPass(),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == (2 * 2 * 2) + 2
        assert PassContext.current().get_trace_stack_size() == 1
        assert PassContext.current().get_current_trace().size == 7


def test_passes_with_mixed_granularities():
    @tvm.script.ir_module
    class MockModule:
        @R.function
        def f1(x: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, x)
                gv0 = relax.add(x, x)
                relax.output(gv0)
            return gv0

        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                gv0 = relax.add(lv0, y)
                relax.output(gv0)
            gv1 = relax.multiply(x, y)
            gv2 = relax.add(gv1, y)
            return (gv0, gv1, gv2)

    mod = MockModule
    assert isinstance(mod, tvm.IRModule)

    # Helper function for tuning
    def pass_func(
        mod: IRModule, ctx: PassContext, eval_passes: List[transform.Pass] = None
    ) -> IRModule:
        trace = ctx.pop_trace()

        # Create mock choices for testing
        choices = [Choice(), Choice(), Choice()]
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockTuningKnob", choices)

        candidates = default_generate_candidate([knob], trace, eval_passes)
        mock_evaluate(candidates, "llvm", ctx)
        best_trace = select_best_candidate(candidates)

        ctx.push_trace(best_trace)
        return best_trace.out_mod

    @ir.transform.module_pass(opt_level=0, traceable=True)
    def MockModulePass(mod: IRModule, ctx: PassContext) -> IRModule:
        # Input granularity == Candidate granularity.
        return pass_func(mod, ctx)

    @relax.transform.function_pass(opt_level=0, traceable=True)
    def MockFunctionPass(func: Expr, mod: IRModule, ctx: PassContext) -> Function:
        # Input granularity > Candidate granularity.
        # Start trace with smaller granularity: IRModule->Function.
        ctx.push_trace(Trace(IRModule.from_expr(func)))
        # Do something.
        pass_func(mod, ctx)
        # Pop tuned trace and recover the previous trace.
        ctx.pop_trace()
        return func

    @relax.transform.dataflowblock_pass(opt_level=0, traceable=True)
    def MockDataflowBlockPass(
        block: DataflowBlock, mod: IRModule, ctx: PassContext
    ) -> DataflowBlock:
        # TODO(sunggg): figure out how to create IRModule from DataflowBlock
        # Provide random binding for now
        x = relax.Var("x", [tvm.tir.Var("n", "int64")], relax.DynTensorType(1, "float32"))
        seq_expr = relax.SeqExpr([block], x)
        ret_type = relax.DynTensorType(-1, "float32")
        func = relax.Function([x], seq_expr, ret_type)
        ctx.push_trace(Trace(IRModule.from_expr(func)))
        # Do something
        pass_func(mod, ctx)
        ctx.pop_trace()
        return block

    seq = transform.Sequential(
        [
            MockModulePass,
            MockFunctionPass,
            MockDataflowBlockPass,
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=[]):
        _ = seq(mod)
        # Trace length and num eval can be different depending on how each function/dataflow block is treated.


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
    # test_choice()
    # test_knob()
    # test_trace()
    # test_default_functions()
    test_pass_context()
    # pytest.main([__file__])
