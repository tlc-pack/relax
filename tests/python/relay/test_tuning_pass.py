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
""" test tuning pass in relay side"""
import logging
import tvm
from tvm import relay, ir
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
import numpy as np
from typing import List
import copy
from tvm.relay.analysis import post_order_visit
from tvm.relay.expr import Call
from tvm.relay.expr_functor import ExprMutator

from tvm.ir.transform import Pass, TuningPass, Trace, Choice, Instruction, Sequential
from tvm.meta_schedule.database import JSONDatabase
import os, shutil


@tvm.instrument.pass_instrument
class PassTracker:
    def run_before_pass(self, module, info):
        print(f"pass name: {info.name}")


CONFIG = {"target": "llvm", "target_host": "llvm", "device_id": 0}


def example(x_shape=(1, 32, 16, 16), channels1=32, channels2=32, channels3=32, channels4=32):
    in_c = x_shape[1]
    x = relay.var("x", shape=x_shape)
    w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
    w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
    w3 = relay.var("w3", shape=(channels3, in_c, 3, 3))
    w4 = relay.var("w4", shape=(channels4, in_c, 1, 1))

    args = [x, w1, w2, w3, w4]
    y1 = relay.nn.conv2d(x, w1)
    y2 = relay.nn.conv2d(x, w2)
    # y3 cannot be combined
    y3 = relay.nn.conv2d(x, w3)
    y4 = relay.nn.conv2d(x, w4)
    y5 = relay.nn.max_pool2d(x)

    c_data = np.empty(x_shape).astype("float32")
    c = relay.const(c_data)
    y6 = relay.add(c, c)
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.multiply(y6, relay.const(13, "float32"))
    y6 = relay.add(y6, y6)

    z = relay.Tuple((y1, y2, y3, y4, y5, y6))

    return relay.Function(args, z)


@ir.transform.module_pass(opt_level=0)
class HeuristicPass(Pass):
    def __init__(self, required: List[Pass] = []):
        super().__init__("HeuristicPass", kind=0, required=required)

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        return relay.transform.FoldConstant()(mod)


@ir.transform.module_pass(opt_level=0)
class TuningLayoutPass(TuningPass):
    def __init__(self, eval_passes=[], required=[], database=None):
        super().__init__(
            "TuneLayout",
            eval_passes=eval_passes,
            required=required,
            database=database,
        )
        self.num_evals = 0

    def tune(self, trace, ctx):
        def apply(mod):
            new_mod = relay.transform.InferType()(mod)
            new_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(new_mod)
            return new_mod

        def noapply(mod):
            return mod

        choices = {"On": Choice(apply), "Off": Choice(noapply)}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Instruction("InstructionTuningLayout", choices)

        candidates = knob.generate_candidates(trace)

        candidates = self.consider_eval_passes(candidates, ctx)
        self.evaluate(ctx, candidates)
        best_trace = self.select_best_candidate(candidates)
        return best_trace

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        best_trace = self.tune(Trace(mod), ctx)
        return best_trace.out_mod


@ir.transform.module_pass(opt_level=0)
class TuningParallelConv2dPass(TuningPass):
    def __init__(self, eval_passes=[], required=[], database=None):
        super().__init__(
            "TuneCombineParallelConv2D",
            eval_passes=eval_passes,
            required=required,
            database=database,
        )

    def tune(self, trace, ctx):
        def apply(mod):
            new_mod = relay.transform.InferType()(mod)
            new_mod = relay.transform.CombineParallelConv2D(min_num_branches=2)(new_mod)
            return new_mod

        def noapply(mod):
            return mod

        choices = {"On": Choice(apply), "Off": Choice(noapply)}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Instruction("InstructionTuningParallelConv2D", choices)

        candidates = knob.generate_candidates(trace)

        candidates = self.consider_eval_passes(candidates, ctx)
        self.evaluate(ctx, candidates)
        best_trace = self.select_best_candidate(candidates)
        return best_trace

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        best_trace = self.tune(Trace(mod), ctx)
        return best_trace.out_mod


@ir.transform.module_pass(opt_level=0)
class TuningSubgraphPass(TuningPass):
    def __init__(self, eval_passes=[], required=[], database=None):
        super().__init__(
            "SubgraphTuningPass",
            eval_passes=eval_passes,
            required=required,
            database=database,
        )

    def tune(self, trace, ctx):
        def subgraph_tuning(mod):
            def convert_conv2d_NHWC(mod):
                new_mod = relay.transform.InferType()(mod)
                new_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(new_mod)
                return new_mod

            def convert_conv2d_NCHW(mod):
                new_mod = relay.transform.InferType()(mod)
                new_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]})(new_mod)
                return new_mod

            def noapply(mod):
                return mod

            choices = {
                "convert_conv2d_NHWC": Choice(convert_conv2d_NHWC),
                "convert_conv2d_NCHW": Choice(convert_conv2d_NCHW),
                "NoApply": Choice(noapply),
            }
            subknob = Instruction("LayoutTransform", choices)

            # Collect nodes to traverse
            lst = []

            def fvisit(node):
                if isinstance(node, Call):
                    lst.append(node)

            expr = mod["main"].body
            post_order_visit(expr, fvisit)

            # Mutator class to apply decisions
            class MyMutator(ExprMutator):
                def __init__(self, decisions):
                    self.decisions = decisions
                    super().__init__()
                    self.cnt = 0

                def visit_call(self, call):
                    fn = self.visit(call.op)
                    args = []
                    for arg in call.args:
                        args.append(self.visit(arg))

                    if call in self.decisions:
                        self.cnt += 1
                        return self.decisions[call].out_mod["main"].body

                    return Call(fn, args, call.attrs)

            # Iterate over nodes and benchmark possible decisions on each subgraph
            subdecisions = dict()
            for node in lst:
                assert isinstance(node, Call)
                if node.op.name == "nn.conv2d":
                    submod = tvm.IRModule.from_expr(node)
                    subcandidates = []
                    for decision in subknob.choices:
                        subtrace = Trace(submod, trace=[(subknob, decision)])
                        subcandidates.append(subtrace)

                    subcandidates = self.consider_eval_passes(subcandidates, ctx)
                    self.evaluate(ctx, subcandidates)
                    best_subtrace = self.select_best_candidate(subcandidates)
                    subdecisions[node] = best_subtrace

            # Apply the best subdecisions
            mutator = MyMutator(subdecisions)
            expr = mutator.visit(expr)
            return tvm.IRModule.from_expr(expr)

        # Higher-level knob wrapping subgraph tuning
        knob = Instruction("InstructionTuningSubgraph", choices=[Choice(subgraph_tuning)])
        best_trace = copy.deepcopy(trace)
        best_trace.add(knob, 0)
        return best_trace

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        best_trace = self.tune(Trace(mod), ctx)

        return best_trace.out_mod


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
        knob = Instruction("MockTuning", choices)

        candidates = knob.generate_candidates(trace)

        candidates = self.consider_eval_passes(candidates, ctx)
        self.evaluate(ctx, candidates)
        best_trace = self.select_best_candidate(candidates)
        return best_trace

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        best_trace = self.tune(Trace(mod), ctx)

        return best_trace.out_mod


# TODO: support ir.Sequential
def apply_sequential(seq, mod):
    cnt = 0
    for p in seq:
        mod = p(mod)
        cnt += p.num_evals
    return mod, cnt


def test_tuning_pass(f=example(), config={"target": "llvm", "target_host": "llvm", "device_id": 0}):
    mod = tvm.IRModule.from_expr(f)
    custom_pass = TuningParallelConv2dPass()
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 2

    mod = tvm.IRModule.from_expr(f)
    custom_pass = MockTuningPass()
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3

    mod = tvm.IRModule.from_expr(f)
    custom_pass = TuningSubgraphPass()
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    # 4 conv * 3 choices each
    assert TuningPass.total_num_evals == 12


# Sequential applies each pass one-by-one
# In this case, search space grows in additive manner
def test_sequential(f=example(), config={"target": "llvm", "target_host": "llvm", "device_id": 0}):

    mod = tvm.IRModule.from_expr(f)
    seq = [MockTuningPass(eval_passes=[]), TuningLayoutPass(eval_passes=[])]
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod, cnt = apply_sequential(seq, mod)

    assert TuningPass.total_num_evals == 3 + 2


# Joint-optimization expands each candidate with its eval_passes
# In this case, search space grows in combinatorial manner
def test_joint_optimization(
    f=example(), config={"target": "llvm", "target_host": "llvm", "device_id": 0}
):
    mod = tvm.IRModule.from_expr(f)
    custom_pass = MockTuningPass(eval_passes=[TuningLayoutPass(eval_passes=[])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 2

    mod = tvm.IRModule.from_expr(f)
    custom_pass = TuningLayoutPass(eval_passes=[MockTuningPass(eval_passes=[])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 2 * 3

    # Heurstic pass does not affect the search space
    mod = tvm.IRModule.from_expr(f)
    custom_pass = TuningLayoutPass(eval_passes=[MockTuningPass(eval_passes=[HeuristicPass()])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 2

    mod = tvm.IRModule.from_expr(f)
    custom_pass = MockTuningPass(eval_passes=[TuningLayoutPass(eval_passes=[MockTuningPass()])])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * 2 * 3

    mod = tvm.IRModule.from_expr(f)
    custom_pass = MockTuningPass(eval_passes=[TuningLayoutPass(), MockTuningPass()])
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    assert TuningPass.total_num_evals == 3 * (2 + 3)

    mod = tvm.IRModule.from_expr(f)
    custom_pass = MockTuningPass(
        eval_passes=[TuningLayoutPass(eval_passes=[MockTuningPass()]), MockTuningPass()]
    )
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)
    assert TuningPass.total_num_evals == 3 * (2 * 3 + 3)


def test_database(f=example(), config=CONFIG, remove_after=False):
    def _create_json_database(tmpdir: str) -> JSONDatabase:
        path_workload = os.path.join(tmpdir, "workloads.json")
        path_tuning_record = os.path.join(tmpdir, "tuning_records.json")
        return JSONDatabase(path_workload, path_tuning_record)

    path = "./tmp"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
    mod = tvm.IRModule.from_expr(f)
    custom_pass = TuningParallelConv2dPass(database=_create_json_database(path))
    TuningPass.total_num_evals = 0
    with PassContext(opt_level=4, config=config):
        mod = custom_pass(mod)

    if remove_after:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


if __name__ == "__main__":
    # test_tuning_pass()
    # test_sequential()
    # test_joint_optimization()
    test_database()
