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
"""Relax Tuning Pass API"""
import tvm
from tvm._ffi import register_object
from tvm.runtime import Object
from . import _ffi_api
from tvm.ir.module import IRModule
from tvm.relax import Expr
from typing import Callable, Union, Dict, List, Optional
from tvm.ir.transform import PassContext, Pass
import copy, sys

from tvm import meta_schedule
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)
from tvm.meta_schedule.database import TuningRecord
import itertools
import numpy as np
from tvm._ffi.registry import register_func

# Default constraint func that always returns true
def f_default_constr(mod: IRModule):
    return True


@register_object("relax.transform.Choice")
class Choice(Object):
    """
    A TVM object Choice that maintains a set of transformation and constraint functions
    Transformation function should be applied when constraint function returns true
    Parameters
    ----------
    f_transform : Callable
        Transformation function

    f_constr : Callable
        Constraint function
    """

    f_transform: Callable
    f_constr: Callable

    def __init__(self, f_transform: Callable, f_constr: Callable = None):
        """Constructor
        Parameters
        ----------
        f_transform : Callable
            Transformation function

        f_constr : Callable
            Constraint function


        Returns
        -------
        ret: Object

        """

        f_constr = f_constr if f_constr else f_default_constr

        self.__init_handle_by_constructor__(
            _ffi_api.Choice, f_transform, f_constr  # type: ignore # pylint: disable=no-member
        )

    def get_transform_func(self) -> Callable:
        """Get f_transform
        Parameters
        ----------
        func : Expr
            The closure, can be ExternFunc or PrimFunc.

        args : Union[Tuple, List[Expr]]
            The input arguments.


        Returns
        -------
        ret: Object

        """
        return _ffi_api.ChoiceGetTransformFunc(self)

    def get_constr_func(self) -> Callable:
        """Get f_constr"""
        return _ffi_api.ChoiceGetConstrFunc(self)

    def check_constr(self, mod: IRModule) -> bool:
        """Check if the given constraint satisfies"""
        return _ffi_api.ChoiceCheckConstr(self, mod)


@register_object("relax.transform.Knob")
class Knob(Object):
    """
    A TVM object Knob to support customization on the python side.
    """

    def __init__(self, name: str, choices: Union[List[Choice], Dict[str, Choice]]):
        """Constructor"""
        if isinstance(choices, list):
            choices = {str(idx): val for idx, val in enumerate(choices)}

        self.__init_handle_by_constructor__(
            _ffi_api.Knob, name, choices  # type: ignore # pylint: disable=no-member
        )

    def verify(self, decision: Union[str, int]) -> bool:
        """Verify if the decision is valid"""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.KnobVerify(self, decision)

    def apply(self, mod: IRModule, decision: Union[str, int]) -> IRModule:
        """Get choice if a decision is valid"""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.KnobApply(self, mod, decision)

    def __str__(self) -> str:
        msg = f"{self.name} (# of choices: {len(self.choices)})\n"
        for name, choice in self.choices.items():
            msg += f"  - {name}: {choice}\n"
        return msg


@register_object("relax.transform.Trace")
class Trace(Object):
    """
    A TVM object Trace to support customization on the python side.
    """

    def __init__(
        self,
        in_mod: IRModule,
        knobs: Optional[List[Knob]] = None,
        decisions: Optional[List[Union[str, int]]] = None,
    ):
        """Constructor"""
        knobs = knobs if knobs else list()
        decisions = (
            [str(v) if isinstance(v, int) else v for v in decisions] if decisions else list()
        )
        self.__init_handle_by_constructor__(
            _ffi_api.Trace, in_mod, knobs, decisions  # type: ignore # pylint: disable=no-member
        )

    def verify(self) -> bool:
        """Verify if current history is valid"""
        return _ffi_api.TraceVerify()

    def add(self, knob: Knob, decision: Union[str, int]) -> IRModule:
        """Add & Apply new decision (with knob)"""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.TraceAdd(self, knob, decision)

    def set_perf(self, perf: float) -> None:
        return _ffi_api.TraceSetPerf(self, perf)

    def __str__(self) -> str:
        n = len(self.knobs)
        msg = f"Trace length: {n}\n"
        for idx in range(n):
            msg += f"[{idx+1}] {self.knobs[idx].name}: {self.decisions[idx]}\n"
        return msg


# Default functions for writing a tuning pass
@register_func("relax.transform.default_generate_candidate")
def default_generate_candidate(
    knob: Knob, trace: Trace, eval_passes: List[Pass] = None
) -> List[Trace]:
    """
    Default function to generate the search space for a given trace by using registered choices
    This function simply expands candidate space as long as the knob's constraint satisfies
    To reduce the search space, a developer may expand each choice with smart search method
    (e.g., genetic search, multi-armed bandit)

    """
    mod = trace.out_mod
    candidates = list()
    # Iterate over every decision
    for decision in knob.choices.keys():
        choice = knob.choices[decision]
        # Generate new candidate when this condition satisfies
        if choice.check_constr(mod):
            new_trace = copy.deepcopy(trace)
            new_trace.add(knob, decision)
            candidates.append(new_trace)

    # Expand candidates by using eval passes if provided
    if eval_passes:
        candidates = default_consider_eval_passes(candidates, eval_passes)
    return candidates


@register_func("relax.transform.default_consider_eval_passes")
def default_consider_eval_passes(
    init_candidates: List[Trace], eval_passes: List[Pass] = None
) -> List[Pass]:
    """
    Expands traces generated by current tuning pass with its eval passes
    This function visits each pass in depth-first manner
    """
    ctx = PassContext.current()
    candidates = list(init_candidates)
    for _ in range(len(candidates)):
        trace = candidates.pop(0)
        ctx.set_trace(trace)
        for eval_pass in eval_passes:
            # Both heuristic and tuning passes can be either traced or not
            # If an eval pass is traced, expands candidates by visiting its evaluation passes in dfs

            # TODO: handle function_pass, dataflowblock_pass
            if eval_pass.info.traceable:
                eval_pass(trace.out_mod)
            # If an eval pass is not traced, create a single choice knob for tracking purpose
            else:

                def non_traceable_pass(mod):
                    return eval_pass(mod)

                knob = Knob(f"{eval_pass.info.name}", [Choice(non_traceable_pass)])
                trace.add(knob, 0)

        # A new trace contains the best decisions in eval_passes
        candidates.append(copy.deepcopy(ctx.trace))

    return candidates


@register_func("relax.transform.default_evaluate")
def default_evaluate(
    candidates: List[Trace],
    target_str: str,
    builder: Optional[meta_schedule.builder.Builder] = None,
    runner: Optional[meta_schedule.runner.Runner] = None,
) -> int:
    ctx = PassContext.current()
    target = tvm.target.Target(target_str)
    # Setup default builder if not provided
    if builder is None:

        def relax_build(
            mod: IRModule,
            target: tvm.target.Target,
            params: dict = {},
        ):
            mod = tvm.relax.transform.BindParams("main", params)(mod)
            ex = tvm.relax.vm.build(mod, target)
            return ex.mod

        builder = LocalBuilder(f_build=relax_build)

    # Setup default runner if not provided
    if runner is None:

        def relax_eval_func(rt_mod, device, evaluator_config, repeated_args):
            exec = tvm.relax.vm.Executable(rt_mod)
            vm = tvm.relax.VirtualMachine(exec=exec, device=device)

            eval = vm.module.time_evaluator(
                func_name="main",
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
            )
            repeated_costs: List[List[float]] = []
            for args in repeated_args:
                profile_result = eval(*args)
                repeated_costs.append(profile_result.results)

            costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]

            return costs

        runner = LocalRunner(
            evaluator_config=EvaluatorConfig(),
            f_run_evaluator=relax_eval_func,
        )

    num_evals = 0
    # Evaluation
    for candidate in candidates:
        # If this candidate is already evaluated, skip the measurement
        if candidate.perf != -1:
            continue

        num_evals += 1

        mock_perf = ctx.num_evals + num_evals
        candidate.set_perf(mock_perf)

        # TODO(sunggg): resolve issue with param binding
        continue
        mod = candidate.out_mod
        # Evaluate candidates
        # Build candidate
        (builder_result,) = builder.build([BuilderInput(mod, target)])

        assert builder_result.artifact_path is not None
        assert builder_result.error_msg is None

        runner_input = RunnerInput(
            builder_result.artifact_path,
            target_str,
            args_info=[
                TensorInfo(shape=[int(i) for i in p.shape], dtype=p.checked_type.dtype)
                for p in mod["main"].params
            ],  # convert list[Var] to list[TensorInfo]
        )
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()

        assert runner_result.error_msg is None
        perfs = []
        for result in runner_result.run_secs:
            if isinstance(result, tvm.tir.FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0
            perfs.append(result)

        # Store the evaluation result
        candidate.update_perf(np.mean(perfs))

        # clean up
        f_clean_build = get_global_func_with_default_on_worker(
            "meta_schedule.remove_build_dir", None
        )
        if f_clean_build is not None:
            f_clean_build(builder_result.artifact_path)
        else:
            raise RuntimeError("Unable to find remove_build_dir function.")

        # TODO(sunggg): Make database work

    ctx.inc_num_evals(num_evals)
    return num_evals


# Choose the best trace
def select_best_candidate(traces):
    best_perf, best_trace = sys.maxsize, None
    for candidate in traces:
        avg = candidate.perf
        # Select best one
        if best_perf > avg:
            best_perf = avg
            best_trace = candidate
    return best_trace


# Return trace wrapper if necessary
def get_trace(in_: Union[Trace, IRModule, Expr]) -> Trace:
    if isinstance(in_, Trace):
        return in_
    if isinstance(in_, IRModule):
        return Trace(in_)
    elif isinstance(in_, Expr):
        return Trace(tvm.IRModule.from_expr(in_))
    else:
        raise Exception(f"Invalid input type for trace: {type(in_)}")


# Extracts matching subgraph for subgraph-level tuning
def extract_subgraph(mod, pattern):
    assert 0, "Need to implement"


# [Optional] a cost model that estimates the performance of a trace
def query_cost_model(cost_model, trace: Trace) -> float:
    assert 0, "Need to implement"


# [Optional] a prediction model that predicts the optimized IRModule
# This can be done by heuristic like AutoTVM
# or data-driven approach like ApplyHistoryBest in MetaSchedule
def predict(mod: IRModule, ctx) -> IRModule:
    assert 0, "Need to implement"
