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

# TODO(sunggg):
# 1) Better Integration with MetaSchedule
#    1-1) Trace with MetaSchedule Trace
#    1-2) Database (includes serialization)
#    1-3) Other componets (e.g., MetaSchedule Instr, Cost model)
# 2) Better example for subgraph-level tuning
#    *** This is currently blocked by pattern matcher, modular compilation, etc. ***

from typing import Callable, Union, Dict, List, Optional
import copy
import sys
import itertools
import logging
import numpy as np
import tvm
from tvm.runtime import Object
from tvm.ir.module import IRModule
from tvm.relax import Expr
from tvm.ir.transform import PassContext, Pass
from tvm import meta_schedule
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)
from tvm._ffi import register_object
from tvm._ffi.registry import register_func
from . import _ffi_api

logger = logging.getLogger("TuningAPI")  # pylint: disable=invalid-name

# Default constraint func that always returns true
def f_default_constr(mod: IRModule):  # pylint: disable=unused-argument
    return True


@register_object("relax.transform.Choice")
class Choice(Object):
    """
    A TVM object Choice that maintains a set of transformation and constraint functions.
    Transformation function should be applied when constraint function returns true.
    Parameters
    ----------
    f_transform : Callable
        Transformation function.

    f_constr : Callable
        Constraint function.

    Examples
    --------
    The following code block defines a Choice.

    .. code-block:: python

        def apply(mod):
            return relax.transform.FoldConstant()(mod)
        def constr(mod):
            return len(mod.functions) == 3
        # Define a choice to apply constant folding only when IRModule has three functions.
        choice = Choice(apply, constr)
    """

    def __init__(self, f_transform: Callable, f_constr: Optional[Callable] = None):
        """Constructor
        Parameters
        ----------
        f_transform : Callable
            Transformation function.

        f_constr : Callable
            Constraint function.
        """
        f_constr = f_constr if f_constr else f_default_constr
        self.__init_handle_by_constructor__(
            _ffi_api.Choice, f_transform, f_constr  # type: ignore # pylint: disable=no-member
        )

    def get_transform_func(self) -> Callable:
        """Getter for f_transform
        Returns
        -------
        ret: Callable
           registered transformation function
        """
        return _ffi_api.ChoiceGetTransformFunc(self)

    def get_constr_func(self) -> Callable:
        """Getter for f_constr
        Returns
        -------
        ret: Callable
           registered constraint function
        """
        return _ffi_api.ChoiceGetConstrFunc(self)

    def check_constr(self, mod: IRModule) -> bool:
        """Perform f_constr
        Returns
        -------
        ret: Bool
           Returns whether the IRModule satisfies the constraint or not
        """
        return _ffi_api.ChoiceCheckConstr(self, mod)


@register_object("relax.transform.Knob")
class Knob(Object):
    """
    A TVM object Knob that maintains a set of valid Choices.
    By using Knobs, a tuning pass can generate candidates and define the search space.
    Parameters
    ----------
    name : str
        Name of the knob.

    choices: Union[List[Choice], Dict[str, Choice]]
        A list of valid choices

    Examples
    --------
    The following code block defines a Knob.

    .. code-block:: python

        def apply(mod):
            return relax.transform.FoldConstant()(mod)
        def noapply(mod):
            return mod
        choices = {"apply": Choice(apply), "noapply": Choice(noapply)}
        # A knob manages a set of its valid choices
        knob = Knob("MockTuningKnob", choices)
    """

    def __init__(self, name: str, choices: Union[List[Choice], Dict[str, Choice]]):
        """Constructor."""
        if isinstance(choices, list):
            choices = {str(idx): val for idx, val in enumerate(choices)}

        self.__init_handle_by_constructor__(
            _ffi_api.Knob, name, choices  # type: ignore # pylint: disable=no-member
        )

    def verify(self, decision: Union[str, int]) -> bool:
        """Verify if the decision is valid."""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.KnobVerify(self, decision)

    def apply(self, mod: IRModule, decision: Union[str, int]) -> IRModule:
        """Get choice if a decision is valid."""
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
    A TVM object Trace logs the history of transformations (decisions).
    Parameters
    ----------
    in_mod : IRModule
        Input IRModule.
    knobs: Optional[List[Knob]]
        A list of knobs applied in the trace.
    decisions: Optional[List[Union[str, int]]]
        A list of decisions made for each knob

    Examples
    --------
    The following code block defines a Trace.

    .. code-block:: python

        trace = Trace(mod, [knob1, knob2, knob3], ["c1", "c0", "c3"])
        assert trace.size == 3 # Length of history.
        # 'out' contains IRModule that applies transformations in the trace.
        out: IRModule = trace.add(knob4, "c2")
        assert trace.size == 4 # Length of history.
        trace.set_perf(0.03) # Set the performance number of the trace.
    """

    def __init__(
        self,
        in_mod: IRModule,
        knobs: Optional[List[Knob]] = None,
        decisions: Optional[List[Union[str, int]]] = None,
    ):
        """Constructor."""
        knobs = knobs if knobs else list()
        decisions = (
            [str(v) if isinstance(v, int) else v for v in decisions] if decisions else list()
        )
        self.__init_handle_by_constructor__(
            _ffi_api.Trace, in_mod, knobs, decisions  # type: ignore # pylint: disable=no-member
        )

    def verify(self) -> bool:
        """Verify if current history is valid."""
        return _ffi_api.TraceVerify()

    def add(self, knob: Knob, decision: Union[str, int]) -> IRModule:
        """Add & Apply new decision (with knob)."""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.TraceAdd(self, knob, decision)

    def set_perf(self, perf: float) -> None:
        """Set performance number for the trace."""
        return _ffi_api.TraceSetPerf(self, perf)

    def __str__(self) -> str:
        n = len(self.knobs)
        msg = f"Trace length: {n}\n"
        for idx in range(n):
            msg += f"[{idx+1}] {self.knobs[idx].name}: {self.decisions[idx]}\n"
        return msg


@register_func("relax.transform.default_generate_candidate")
def default_generate_candidate(
    knobs: List[Knob], trace: Trace, eval_passes: Optional[List[Pass]] = None
) -> List[Trace]:
    """
    Default function to generate the search space for a given trace by using registered choices.
    This function simply expands candidate space as long as the knob's constraint satisfies.
    To reduce the search space, a developer may expand each choice with smart search method.
    (e.g., genetic search, multi-armed bandit)
    Note, each pass generates candidates without worrying about the interaction with other passes.
    i.e., it only uses its incoming trace/IRModule and Choices for candidate generation.
    This will help alleviating the complexity of joint-optimization significantly.
    - consideration of interaction between optimizations has known to be extremely difficult.

    Parameters
    ----------
    knobs : List[Knob]
        List of Knobs to consider to generate candidate for input trace.
    trace: Trace
        Input trace.
    eval_passes: Optional[List[Pass]]
        List of passes to consider to evaluate each candidate.
        This will enable joint-optimization.

    Return
    ----------
    candidates: List[Trace]
        List of candidate traces
    """

    candidates = [trace]
    # Iterate over every decision
    for knob in knobs:
        num = len(candidates)
        for _ in range(num):
            cur_trace = candidates.pop(0)
            for decision in knob.choices.keys():
                choice = knob.choices[decision]
                # Generate new candidate when this condition satisfies.
                if choice.check_constr(cur_trace.out_mod):
                    new_trace = copy.deepcopy(cur_trace)
                    new_trace.add(knob, decision)
                    candidates.append(new_trace)

    # Expand candidates by using eval passes if provided. This will enable joint-optimization.
    if eval_passes:
        candidates = default_consider_eval_passes(candidates, eval_passes)
    return candidates


@register_func("relax.transform.default_consider_eval_passes")
def default_consider_eval_passes(
    init_candidates: List[Trace], eval_passes: Optional[List[Pass]] = None
) -> List[Trace]:
    """
    Default function to update traces with eval passes.
    It visits each eval_pass in dfs order in transform.Sequential() and
    returns the best possible candidate trace for each candidate.

    Parameters
    ----------
    init_candidates: List[Trace]
        Initial candidates
    eval_passes: Optional[List[Pass]]
        List of passes to consider to evaluate each candidate.
        This will enable joint-optimization.
    Return
    ----------
    candidates: List[Trace]
        List of candidate traces
    """
    if not eval_passes:
        return init_candidates

    eval_passes = list(eval_passes) if not isinstance(eval_passes, list) else eval_passes
    ctx = PassContext.current()
    candidates = []
    # for _ in range(len(candidates)):
    for trace in init_candidates:
        ctx.push_trace(trace)
        tvm.transform.Sequential(eval_passes)(trace.out_mod)
        new_trace = ctx.pop_trace()
        # A new trace contains the best decisions in eval_passes
        candidates.append(new_trace)

    return candidates


@register_func("relax.transform.default_evaluate")
def default_evaluate(
    candidates: List[Trace],
    target_str: str,
    params: Optional[Dict[str, np.ndarray]] = None,
    builder: Optional[meta_schedule.builder.Builder] = None,
    runner: Optional[meta_schedule.runner.Runner] = None,
) -> None:
    """
    Default function to evaluate a set of candidate traces by using MetaSchedule builder/runner.

    Parameters
    ----------
    candidates: List[Trace]
        List of traces to evaluate.
    target_str: str,
        Compilation target (e.g., llvm, cuda).
    params: Optional[Dict[str, np.ndarray]]
        Params to bind.
    builder: Optional[meta_schedule.builder.Builder]
        builder function. If not provided, default local builder will be used.
    runner: Optional[meta_schedule.runner.Runner]
        runner function. If not provided, default local runner will be used.
    """

    ctx = PassContext.current()
    target = tvm.target.Target(target_str)
    # Setup default local builder if not provided
    if builder is None:

        def relax_build(
            mod: IRModule,
            target: tvm.target.Target,
            params: Optional[Dict[str, np.ndarray]],
        ):
            if params:
                mod = tvm.relax.transform.BindParams("main", params)(mod)
            relax_exec = tvm.relax.vm.build(mod, target)
            return relax_exec.mod

        builder = LocalBuilder(f_build=relax_build)

    # Setup default local runner if not provided
    if runner is None:

        def relax_eval_func(rt_mod, device, evaluator_config, repeated_args):
            relax_exec = tvm.relax.vm.Executable(rt_mod)
            relax_vm = tvm.relax.VirtualMachine(exec=relax_exec, device=device)

            evaluator = relax_vm.module.time_evaluator(
                func_name="main",
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
            )
            repeated_costs: List[List[float]] = []
            for args in repeated_args:
                profile_result = evaluator(*args)
                repeated_costs.append(profile_result.results)

            costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]

            return costs

        runner = LocalRunner(
            evaluator_config=EvaluatorConfig(),
            f_run_evaluator=relax_eval_func,
        )

    # set up clean up function
    f_clean_build = get_global_func_with_default_on_worker("meta_schedule.remove_build_dir", None)
    assert f_clean_build

    # Keep track of number of evaluations (mostly for the debugging purpose)
    num_evals = 0
    # Evaluation
    for candidate in candidates:
        # If this candidate is already evaluated, skip the measurement
        if candidate.perf != -1:
            continue

        # Evaluate candidates
        num_evals += 1
        mod = candidate.out_mod
        # Build candidate
        (builder_result,) = builder.build([BuilderInput(mod, target, params)])

        # Build error
        # Assign the worst performance and move on to the next candidate.
        if builder_result.artifact_path is None:
            logger.warning(builder_result.error_msg)
            candidate.set_perf(1e100)
            continue

        # If build passes, set up runner input and measure the performance.
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

        # Runtime error
        # Assign the worst performance and move on to the next candidate.
        if runner_result.error_msg is not None:
            logger.warning(runner_result.error_msg)
            candidate.set_perf(1e100)
        # For valid measurments, compute the average and update the trace performance.
        else:
            perfs = []
            for result in runner_result.run_secs:
                if isinstance(result, tvm.tir.FloatImm):
                    result = result.value
                assert isinstance(result, float)
                assert result >= 0.0
                perfs.append(result)

            # Store the evaluation result
            # TODO(sunggg): Match with MetaSchedule
            candidate.set_perf(np.mean(perfs))

        # Clean up the artifact
        f_clean_build(builder_result.artifact_path)

    ctx.inc_num_evals(num_evals)


def select_best_candidate(candidates: List[Trace]) -> Trace:
    """
    Select the best trace.

    Parameters
    ----------
    candidates: List[Trace]
        Candidate traces

    Return
    ----------
    best_trace: Trace
        Trace with the best performance
    """
    best_perf, best_trace = sys.maxsize, None
    for candidate in candidates:
        avg = candidate.perf
        # Select best one
        if best_perf > avg:
            best_perf = avg
            best_trace = candidate
    return best_trace


def get_trace(in_: Union[Trace, IRModule, Expr]) -> Trace:
    """
    Getter for a trace wrapper.

    Parameters
    ----------
    in_: Union[Trace, IRModule, Expr]
        Input entity
    Return
    ----------
    wrapped: Trace
        Traced entity
    """
    if isinstance(in_, Trace):
        return in_
    if isinstance(in_, IRModule):
        return Trace(in_)
    if isinstance(in_, Expr):
        return Trace(tvm.IRModule.from_expr(in_))

    raise Exception(f"Invalid input type for trace: {type(in_)}")
