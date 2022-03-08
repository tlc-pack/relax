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

from cv2 import transform
import tvm
from tvm import relay, ir
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.relay import Function
from tvm.relay import testing
import numpy as np
from tvm.contrib import graph_executor as runtime
import sys
from tvm.target import Target
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)
from tvm.tir import FloatImm
from tvm.runtime import Module
import itertools
from typing import List, Dict, Callable, Union, Tuple
import copy


@tvm.instrument.pass_instrument
class PassTracker:
    def run_before_pass(self, module, info):
        print(f"pass name: {info.name}")


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


# Knob manages a set of optimization choices. Although it can apply certain decision, its object does not maintain a decision.
class Knob:
    def __init__(
        self, name: str, choices: Union[Dict[str, Callable], Dict[int, Callable], List[Callable]]
    ):
        self.name = name
        self.choices = choices

    def verify(self, decision):
        if isinstance(self.choices, dict):
            return decision in self.choices
        elif isinstance(self.choices, List):
            return decision < len(self.choices)
        else:
            raise Exception("Invalid type for choices")

    def get_choice(self, decision: Union[str, int]):
        assert self.verify(decision)
        return self.choices[decision]

    def apply(self, mod, decision):
        assert self.verify(decision)
        return self.choices[decision](mod)

    def __str__(self):
        msg = f"{self.name} (# of choices: {len(self.choices)})\n"
        if isinstance(self.choices, dict):
            for name, choice in self.choices.items():
                msg += f"  - {name}: {choice}\n"
        elif isinstance(self.choices, List):
            for idx, choice in enumerate(self.choices):
                msg += f"  - {idx}: {choice}\n"
        else:
            raise Exception("Invalid type for choices")
        return msg


# Trace maintains a sequence of knobs and their decisions.
# It maintains the input/output IRModule
class Trace:
    def __init__(self, in_mod: IRModule, trace: List[Tuple[Knob, Union[str, int]]] = []):
        self.in_mod = in_mod
        self.trace = trace
        self.out_mod = self.apply(in_mod, trace)

    def verify(self):
        for (knob, decision) in self.trace:
            if not knob.verify(decision):
                return False
        return True

    def apply(self, in_mod, trace):
        out_mod = in_mod
        for knob, decision in trace:
            if not knob.verify(decision):
                raise Exception("Illegal decision in the trace")
            out_mod = knob.apply(in_mod, decision)
        return out_mod

    def add(self, knob: Knob, decision: Union[str, int]):
        self.out_mod = knob.apply(self.out_mod, decision)
        self.trace.append((knob, decision))

    def __str__(self):
        msg = f"Trace length: {len(self.trace)}\n"
        for idx, (knob, decision) in enumerate(self.trace):
            msg += f"[{idx+1}] {knob.name}: {decision}\n"
        return msg


# Tuning pass can apply transformation by using a set of knobs and record its sequence.
class Pass:
    def __init__(self, name: str, kind: int, required=[]):
        self.name = name
        # NOTE: This is temporary.
        self.kind = kind
        self.required = required

    # Does this pass generate valid IRModule?
    def validate():
        pass


@ir.transform.module_pass(opt_level=0)
class MyHeuristicPass(Pass):
    def __init__(self, required: List[Pass] = []):
        super().__init__("HeuFoldConstant", kind=0, required=required)

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        return relay.transform.FoldConstant()(mod)


class TuningPass(Pass):
    # debugging: static variable to keep track of # of evals
    cnt = 0

    def __init__(self, name: str, eval_passes: List[Pass], required: List[Pass]):
        super().__init__(name, kind=1, required=required)
        self.eval_passes = eval_passes

    # Each tuning pass needs to implement how it generates search space with its knobs
    def generate_candidates(self, trace: Trace) -> List[Trace]:
        assert 0, "Need to implement"

    def consider_eval_passes(
        self, seeds: List[Trace], eval_passes: List[Pass] = None
    ) -> List[Trace]:
        # If not provided, use the default passes
        if eval_passes is None:
            eval_passes = self.eval_passes

        candidates = list(seeds)
        for eval_pass in eval_passes:
            num = len(candidates)
            for i in range(num):
                trace = candidates.pop(0)
                # Heuristic pass creates a wrapping knob, add to the trace, and push new trace
                if eval_pass.kind == 0:
                    knob = Knob(f"{eval_pass.name}", [eval_pass])
                    trace.add(knob, 0)
                    candidates.append(trace)
                # Tuning pass expands candidates by visiting its evaluation passes
                else:
                    traces = eval_pass.generate_candidates_with_eval_passes(trace)
                    candidates.extend(traces)
        return candidates

    def generate_candidates_with_eval_passes(self, trace, eval_passes=None):
        seeds = self.generate_candidates(trace)
        candidates = self.consider_eval_passes(seeds)
        return candidates

    def evaluate(self, ctx, candidates: List[Trace], num: int = 20, repeat: int = 20):
        # These targets will be retrieved from the ctx
        target, dev = "llvm", tvm.cpu()
        scoreboard: Dict[IRModule, float] = dict()

        # Evaluation
        for candidate in candidates:
            mod = candidate.out_mod
            # Evaluate candidates
            def _build(
                mod: Module,
                target: Target,
                params: dict = {},
            ):
                return tvm.relay.build_module._build_module_no_factory(mod, target, "llvm", params)

            TuningPass.cnt += 1
            # Build candidate
            builder = LocalBuilder(f_build=_build)
            (builder_result,) = builder.build([BuilderInput(mod, Target(target))])

            assert builder_result.artifact_path is not None
            assert builder_result.error_msg is None

            # print(type(transformed_candidate["main"]))
            # inputs = relay.analysis.free_vars(transformed_candidate["main"])

            runner_input = RunnerInput(
                builder_result.artifact_path,
                target,
                [
                    # TensorInfo("float32", (MATMUL_N, MATMUL_N)),
                ],
            )

            evaluator_config = EvaluatorConfig(
                number=10,
                repeat=10,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )

            # Wrap with a executor and evaluator configs
            def eval_func(rt_mod, device, evaluator_config, repeated_args):
                rt_mod = tvm.contrib.graph_executor.GraphModule(rt_mod["default"](device))

                eval = rt_mod.module.time_evaluator(
                    func_name="run",
                    dev=device,
                    number=evaluator_config.number,
                    repeat=evaluator_config.repeat,
                    min_repeat_ms=evaluator_config.min_repeat_ms,
                    f_preproc="cache_flush_cpu_non_first_arg"
                    if evaluator_config.enable_cpu_cache_flush
                    else "",
                )
                repeated_costs: List[List[float]] = []
                for args in repeated_args:
                    profile_result = eval(*args)
                    repeated_costs.append(profile_result.results)

                costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
                return costs

            runner = LocalRunner(
                timeout_sec=100,
                evaluator_config=evaluator_config,
                f_run_evaluator=eval_func,
            )

            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()

            assert runner_result.error_msg is None
            perfs = []
            for result in runner_result.run_secs:
                if isinstance(result, FloatImm):
                    result = result.value
                assert isinstance(result, float)
                assert result >= 0.0
                perfs.append(result)

            def _clean_build(artifact_path: str) -> None:
                f_clean_build = get_global_func_with_default_on_worker(
                    "meta_schedule.remove_build_dir", None
                )
                if f_clean_build is not None:
                    f_clean_build(artifact_path)
                else:
                    raise RuntimeError("Unable to find remove_build_dir function.")

            _clean_build(builder_result.artifact_path)

            # Store transformed candidate
            scoreboard[candidate] = tuple([np.mean(perfs), np.std(perfs)])
            print(f"{candidate}")
            print(f"{mod}: {np.mean(perfs)}ms\n\n")

        return scoreboard

    @staticmethod
    def query_cost_model(candidates):
        pass

    # Predict optimized IRModule.
    # This can be done by heuristic like AutoTVM or data-driven approach based on the tuning records.
    @staticmethod
    def predict(mod):
        pass

    # Extracts matching subgraph
    def extract_subgraph(mod, pattern):
        pass

    def select_best_candidate(self, scoreboard):
        best_perf, best_trace = sys.maxsize, None
        for candidate, (avg, std) in scoreboard.items():
            # Select best one
            if best_perf > avg:
                best_perf = avg
                best_trace = candidate
        return best_perf, best_trace


@ir.transform.module_pass(opt_level=0)
class MyTuningPass1(TuningPass):
    def __init__(self, eval_passes=[], required=[]):
        super().__init__("TuneLayoutPass", eval_passes=eval_passes, required=required)

    def generate_candidates(self, trace):
        def apply(mod):
            new_mod = relay.transform.InferType()(mod)
            new_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(new_mod)
            return new_mod

        def noapply(mod):
            return mod

        choices = {"On": apply, "Off": noapply}
        knob = Knob("MyTuningPass1 - Knob: LayoutTransform", choices)

        candidates = list()
        for decision in choices.keys():
            new_trace = copy.deepcopy(trace)
            new_trace.add(knob, decision)
            candidates.append(new_trace)

        return candidates

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        candidates = self.generate_candidates_with_eval_passes(Trace(mod))
        scoreboard = self.evaluate(ctx, candidates)
        best_perf, best_trace = self.select_best_candidate(scoreboard)
        return best_trace.out_mod


@ir.transform.module_pass(opt_level=0)
class MyTuningPass2(TuningPass):
    def __init__(self, eval_passes=[], required=[]):
        super().__init__(
            "TuneCombineParallelConv2DPass",
            eval_passes=eval_passes,
            required=required,
        )

    def generate_candidates(self, trace):
        def apply(mod):
            new_mod = relay.transform.InferType()(mod)
            new_mod = relay.transform.CombineParallelConv2D(min_num_branches=2)(new_mod)
            return new_mod

        def noapply(mod):
            return mod

        choices = {"On": apply, "Off": noapply}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MyTuningPass2 - Knob: CombineParallelConv2D", choices)

        candidates = list()
        for decision in choices.keys():
            new_trace = copy.deepcopy(trace)
            new_trace.add(knob, decision)
            candidates.append(new_trace)

        return candidates

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        # Candidate generation
        candidates = self.generate_candidates_with_eval_passes(Trace(mod))
        scoreboard = self.evaluate(ctx, candidates)
        best_perf, best_trace = self.select_best_candidate(scoreboard)

        print("==== optimized ===")
        print(best_trace)
        print(best_trace.out_mod)
        return best_trace.out_mod


f = example()
mod = tvm.IRModule.from_expr(f)

# Enable joint optimization
custom_pass = MyTuningPass2(
    eval_passes=[
        MyTuningPass1(eval_passes=[MyHeuristicPass()]),
        MyTuningPass1(eval_passes=[]),
    ]
)
mod = custom_pass(mod)

print(f"# of total evaluation: {TuningPass.cnt}")
