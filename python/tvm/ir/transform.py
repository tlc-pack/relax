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
# pylint: disable=invalid-name,unused-argument
"""Common pass infrastructure across IR variants."""
from __future__ import annotations
import types
import inspect
import functools
from xmlrpc.client import Boolean

import tvm._ffi
import tvm.runtime

from . import _ffi_transform_api

# Dependency for tuning pass
from tvm.runtime import Module
import itertools
from typing import Optional, List, Dict, Callable, Union, Tuple
import copy
from .module import IRModule
import numpy as np
import sys
import random


@tvm._ffi.register_object("transform.PassInfo")
class PassInfo(tvm.runtime.Object):
    """The class contains the meta data required by a pass. It is the
    container of information needed by running an optimization or analysis.
    This class can be extended by adding new members when more meta data is
    needed.

    Parameters
    ----------
    opt_level : int
        The optimization level of this pass.

    name : str
        The pass name.

    required : List[str]
        The list of passes that are required by a certain pass.
    """

    def __init__(self, opt_level, name, required=None):
        self.__init_handle_by_constructor__(_ffi_transform_api.PassInfo, opt_level, name, required)


@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    """The basis where a Relay optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the optimization, etc.

    opt_level : Optional[int]
        The optimization level of this pass.

    required_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are required by a certain pass.

    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are disabled.

    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.

    config : Optional[Dict[str, Object]]
        Additional configurations for specific passes.
    """

    def __init__(
        self,
        opt_level=2,
        required_pass=None,
        disabled_pass=None,
        instruments=None,
        config=None,
    ):
        required = list(required_pass) if required_pass else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("required_pass is expected to be the type of " + "list/tuple/set.")

        disabled = list(disabled_pass) if disabled_pass else []
        if not isinstance(disabled, (list, tuple)):
            raise TypeError("disabled_pass is expected to be the type of " + "list/tuple/set.")

        instruments = list(instruments) if instruments else []
        if not isinstance(instruments, (list, tuple)):
            raise TypeError("instruments is expected to be the type of " + "list/tuple/set.")

        config = config if config else None
        self.__init_handle_by_constructor__(
            _ffi_transform_api.PassContext, opt_level, required, disabled, instruments, config
        )

    def __enter__(self):
        _ffi_transform_api.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_transform_api.ExitPassContext(self)

    def override_instruments(self, instruments):
        """Override instruments within this PassContext.

        If there are existing instruments, their ``exit_pass_ctx`` callbacks are called.
        Then switching to new instruments and calling new ``enter_pass_ctx`` callbacks.

        instruments : Sequence[PassInstrument]
            The list of pass instrument implementations.
        """
        _ffi_transform_api.OverrideInstruments(self, instruments)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _ffi_transform_api.GetCurrentPassContext()

    @staticmethod
    def list_configs():
        """List all registered `PassContext` configuration names and metadata.

        Returns
        -------
        configs : Dict[str, Dict[str, str]]

        """
        return _ffi_transform_api.ListConfigs()


@tvm._ffi.register_object("transform.Pass")
class Pass(tvm.runtime.Object):
    """The base class of all passes. All methods here are just simple wrappers
    that are implemented in the backend. They are defined for users to
    conveniently interact with the base class.
    """

    def __init__(self, name: str, kind: int, required=[]):
        self.name = name
        # TODO: This is temporary.
        self.kind = kind
        self.num_evals = 0
        super().__init__()

    @property
    def info(self):
        """Get the pass meta."""
        return _ffi_transform_api.Info(self)

    def __call__(self, mod):
        """Execute the pass. Note that for sequential pass, the dependency among
        different passes will be resolved in the backend.

        Parameters
        ----------
        mod : tvm.IRModule
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm.IRModule
            The updated module after applying this pass.
        """
        return _ffi_transform_api.RunPass(self, mod)


@tvm._ffi.register_object("transform.ModulePass")
class ModulePass(Pass):
    """A pass that works on tvm.IRModule. Users don't need to interact with
    this class directly. Instead, a module pass should be created through
    `module_pass`, because the design of the `module_pass` API is flexible
    enough to handle the creation of a module pass in different manners. In
    addition, all members of a module pass can be accessed from the base class.
    The same rule applies to FunctionPass as well.
    """


# Instruction manages a set of optimization choices. Although it can apply certain decision, its object does not maintain a decision.
# TODO: We may probabilistically extend the search space at each tuning pass
class Choice:
    def __init__(self, func: Callable, prob: float = 1.0, constr=None):
        self.func = func
        self.prob = prob
        # Default constraint always returns true.
        def default_constr():
            return True

        self.constr = default_constr
        if constr is not None:
            self.constr = constr


# TODO: Instruction soudns bit too generic
class Instruction:
    def __init__(
        self, name: str, choices: Union[List[Choice], Dict[str, Choice], Dict[int, Choice]]
    ):
        self.name = name
        self.choices = choices

    def verify(self, decision: Union[str, int]) -> Boolean:
        if isinstance(self.choices, dict):
            return decision in self.choices
        elif isinstance(self.choices, List):
            return decision < len(self.choices)
        else:
            raise Exception("Invalid type for choices")

    def get_choice(self, decision: Union[str, int]) -> Choice:
        assert self.verify(decision)
        return self.choices[decision]

    def apply(self, mod: IRModule, decision: Union[str, int]) -> IRModule:
        assert self.verify(decision)
        return self.choices[decision].func(mod)

    def generate_candidates(self, trace: Trace) -> List[Trace]:
        candidates = list()
        for decision in self.choices.keys():
            choice = self.choices[decision]
            # Generate new candidate when this condition satisfies
            if choice.constr and choice.prob >= random.uniform(0, 1):
                new_trace = copy.deepcopy(trace)
                new_trace.add(self, decision)
                candidates.append(new_trace)
        return candidates

    def __str__(self) -> str:
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
    def __init__(self, in_mod: IRModule, trace: List[Tuple[Instruction, Union[str, int]]] = []):
        self.in_mod = in_mod
        self.trace = trace
        self.out_mod = self.apply(in_mod, trace)
        self.perf = None

    def verify(self):
        for (knob, decision) in self.trace:
            if not knob.verify(decision):
                return False
        return True

    def apply(self, in_mod: IRModule, trace: Trace) -> IRModule:
        out_mod = copy.deepcopy(in_mod)
        for knob, decision in trace:
            if not knob.verify(decision):
                raise Exception("Illegal decision in the trace")
            out_mod = knob.apply(in_mod, decision)
        self.perf = None
        return out_mod

    def add(self, knob: Instruction, decision: Union[str, int]) -> None:
        self.out_mod = knob.apply(self.out_mod, decision)
        self.trace.append((knob, decision))
        self.perf = None

    def __str__(self) -> str:
        msg = f"Trace length: {len(self.trace)}\n"
        for idx, (knob, decision) in enumerate(self.trace):
            msg += f"[{idx+1}] {knob.name}: {decision}\n"
        return msg


class TuningPass(Pass):
    # @sunggg: Debugging. static variable
    total_num_evals = 0

    def __init__(
        self,
        name: str,
        eval_passes: List[Pass] = [],
        required: List[Pass] = [],
        database: Optional[tvm.meta_schedule.database.Database] = None,
    ):
        super().__init__(name, kind=1, required=required)
        self.eval_passes = eval_passes
        self.database = database
        # For debugging
        self.num_evals = 0

    # Each tuning pass needs to implement how it generates search space with its knobs
    def tune(self, trace: Trace, ctx: PassContext) -> List[Trace]:
        assert 0, "Need to implement"

    def consider_eval_passes(
        self, seeds: List[Trace], ctx: PassContext, eval_passes: List[Pass] = None
    ) -> List[Trace]:
        # If not provided, use the default passes
        if eval_passes is None:
            eval_passes = self.eval_passes

        candidates = list(seeds)
        num = len(candidates)
        for i in range(num):
            trace = candidates.pop(0)
            for eval_pass in eval_passes:
                # For heuristic pass, we create an know with single choice for tracking
                if eval_pass.kind == 0:
                    knob = Instruction(f"{eval_pass.name}", [Choice(eval_pass)])
                    trace.add(knob, 0)
                # Tuning pass expands candidates by visiting its evaluation passes
                else:
                    trace = eval_pass.tune(trace, ctx)

            candidates.append(trace)
        return candidates

    def tune_with_eval_passes(self, trace: Trace, ctx: PassContext, eval_passes: List[Pass] = None):
        seeds = self.tune(trace, ctx)
        candidates = self.consider_eval_passes(seeds, ctx)
        return candidates

    def evaluate(self, ctx, candidates: List[Trace], num: int = 20, repeat: int = 20):
        # TODO: Temporary solution to avoid circular dependency
        # from tvm.meta_schedule.arg_info import TensorInfo
        from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
        from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
        from tvm.meta_schedule.runner import (
            EvaluatorConfig,
            LocalRunner,
            RunnerInput,
        )
        from tvm.meta_schedule.database import TuningRecord

        # These targets will be retrieved from the ctx
        target_str, target_host, device_id = (
            ctx.config["target"],
            ctx.config["target_host"],
            ctx.config["device_id"],
        )
        target = tvm.target.Target(target_str)
        device = tvm.device(target_str, device_id)

        num_evals = 0
        # Evaluation
        for candidate in candidates:
            if candidate.perf is not None:
                continue

            num_evals += 1
            mod = candidate.out_mod
            # Evaluate candidates
            def _build(
                mod: Module,
                target: tvm.target.Target,
                params: dict = {},
            ):
                return tvm.relay.build_module._build_module_no_factory(
                    mod, target, target_host, params
                )

            # Build candidate
            builder = LocalBuilder(f_build=_build)
            (builder_result,) = builder.build([BuilderInput(mod, target)])

            assert builder_result.artifact_path is not None
            assert builder_result.error_msg is None

            runner_input = RunnerInput(
                builder_result.artifact_path,
                target_str,
                [],  # ArgInfo
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
                if isinstance(result, tvm.tir.FloatImm):
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
            # TODO: Replace it with database
            candidate.perf = tuple([np.mean(perfs), np.std(perfs)])

            from tvm.tir import Schedule

            if self.database is not None:
                # TODO: Unify with MetaSchedule trace
                workload = self.database.commit_workload(mod)
                record = TuningRecord(
                    Schedule(mod).trace,
                    perfs,
                    workload,
                    target,
                    [],  # pylint: disable=unsubscriptable-object
                )
                self.database.commit_tuning_record(record)

        TuningPass.total_num_evals += num_evals

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

    def select_best_candidate(self, traces):
        best_perf, best_trace = sys.maxsize, None
        for candidate in traces:
            (avg, std) = candidate.perf
            # Select best one
            if best_perf > avg:
                best_perf = avg
                best_trace = candidate
        return best_trace


@tvm._ffi.register_object("transform.Sequential")
class Sequential(Pass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Note that users can also provide a series of passes that they don't want to
    apply when running a sequential pass. Pass dependency will be resolved in
    the backend as well.

    Parameters
    ----------
    passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    opt_level : Optional[int]
        The optimization level of this sequential pass.
        The opt_level of a default sequential pass is set to 0.
        Note that some of the passes within the Sequantial may still not be executed
        if their opt_level is higher than the provided opt_level.

    name : Optional[str]
        The name of the sequential pass.

    required : Optional[List[str]]
        The list of passes that the sequential pass is dependent on.
    """

    def __init__(self, passes=None, opt_level=0, name="sequential", required=None):
        passes = passes if passes else []
        if not isinstance(passes, (list, tuple)):
            raise TypeError("passes must be a list of Pass objects.")

        required = required if required else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("Required is expected to be the type of list/tuple.")

        self.__init_handle_by_constructor__(
            _ffi_transform_api.Sequential, passes, opt_level, name, required
        )


def _wrap_class_module_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyModulePass(ModulePass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(mod, ctx):
                return inst.transform_module(mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_transform_api.MakeModulePass, _pass_func, pass_info
            )
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyModulePass.__init__, pass_cls.__init__)
    PyModulePass.__name__ = pass_cls.__name__
    PyModulePass.__doc__ = pass_cls.__doc__
    PyModulePass.__module__ = pass_cls.__module__
    return PyModulePass


def module_pass(pass_func=None, opt_level=None, name=None, required=None):
    """Decorate a module pass.

    This function returns a callback when pass_func is provided.
    Otherwise, it serves a decorator function.

    pass_func can also be a class type with a method transform_module.
    This function will create a decorated ModulePass using transform_module
    as the pass function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Module, PassContext) ->Module]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the module pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the module pass is dependent on.

    Returns
    -------
    create_module_pass : Union[Callable, ModulePass]
        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new ModulePass will be returned when we decorate a pass function.
        A new ModulePass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a module pass class.

    .. code-block:: python

        @relay.transform.module_pass
        class CustomPipeline:
            def __init__(self, enable_fold):
                self.enable_fold = enable_fold
                self.cse = relay.transform.EliminateCommonSubexpr()
                self.const_fold = relay.transform.FoldConstant()

            def transform_module(self, mod, ctx):
                mod = self.cse(mod, ctx)
                if self.enable_fold:
                    mod = self.const_fold(mod, ctx)
                return mod

        # create an instance of customized pipeline
        pipeline = CustomPipeline(enable_fold=False)
        assert isinstance(pipeline, transform.ModulePass)
        # run the pipeline.
        output_module = pipeline(input_module)

    The following code creates a module pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relay.transform.module_pass(opt_level=2)
        def transform(mod, ctx):
            tp = relay.TensorType((10,), "float32")
            x = relay.var("x", tp)
            gv = relay.GlobalVar("var")
            func = relay.Function([x], relay.abs(x))
            new_mod = tvm.IRModule({gv: func})
            new_mod.update(mod)
            return new_mod

        module_pass = transform
        assert isinstance(module_pass, transform.ModulePass)
        assert module_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = module_pass(m)
        # Now a function abs should be added to the module m.
    """
    if opt_level is None:
        raise ValueError("Please provide opt_level for the module pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_module_pass(pass_arg):
        """Internal function that creates a module pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_module_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_transform_api.MakeModulePass(pass_arg, info)

    if pass_func:
        return create_module_pass(pass_func)
    return create_module_pass


def PrintIR(header="", show_meta_data=False):
    """A special trace pass that prints the header and IR.

    Parameters
    ----------
    header : str
        The header to be displayed along with the dump.

    show_meta_data : bool
        A boolean flag to indicate if meta data should be printed.

    Returns
    --------
    The pass
    """
    return _ffi_transform_api.PrintIR(header, show_meta_data)
