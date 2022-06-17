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
"""Relax Tuning Pass API primitives"""

from typing import Callable, Union, Dict, List, Optional
import logging
import tvm
from tvm.runtime import Object
from tvm.ir.module import IRModule
from tvm.relax import Expr
from tvm.tir.schedule.trace import JSON_TYPE, _json_from_tvm
from tvm._ffi import register_object
from . import _ffi_api

logger = logging.getLogger("TuningAPI")  # pylint: disable=invalid-name


@register_object("relax.tuning_api.Choice")
class Choice(Object):
    """
    A TVM object Choice that maintains a set of transformation and constraint function keys.
    Corresponding functions should be registered as PackedFunc with these keys.
    Transformation function will be applied when constraint function returns true.
    Parameters
    ----------
    f_transform_key : Optional[str]
        Key for transformation function.
    f_transform_args : Optional[List]
        Arguments for transformation function.
    f_constr_key : Optional[str]
        Key for constraint function.
    f_constr_args : Optional[List]
        Arguments for constraint function.

    Examples
    --------
    The following code block defines a Choice.

    .. code-block:: python
        @tvm.register_func("relax.tuning_api.test.f_transform")
        def apply(mod):
            return relax.tuning_api.FoldConstant()(mod)
        @tvm.register_func("relax.tuning_api.test.f_constr")
        def constr(mod):
            return len(mod.functions) == 3
        # Define a choice to apply constant folding only when IRModule has three functions.
        choice = Choice(f_transform_key = "relax.tuning_api.test.f_transform", f_constr_key = "relax.tuning_api.test.f_constr")
    """

    def __init__(
        self,
        f_transform_key: Optional[str] = None,
        f_transform_args: Optional[List] = None,
        f_constr_key: Optional[str] = None,
        f_constr_args: Optional[List] = None,
    ):
        """Constructor
        Parameters
        ----------
        f_transform_key : Optional[str]
            Key for transformation function.

        f_tramsform_args: Optional[List]
            Arguments for transformation function.

        f_constr_key : Optional[str]
            Key for constraint function.

        f_constr_args: Optional[List]
            Arguments for constraint function.
        """

        if f_transform_key is None:
            f_transform_key = "relax.tuning_api.Choice.f_default_transform"

        if f_transform_args is None:
            f_transform_args = []

        if f_constr_key is None:
            f_constr_key = "relax.tuning_api.Choice.f_default_constr"

        if f_constr_args is None:
            f_constr_args = []

        self.__init_handle_by_constructor__(
            _ffi_api.Choice,
            f_transform_key,
            f_transform_args,
            f_constr_key,
            f_constr_args,  # type: ignore # pylint: disable=no-member
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

    def apply_transform_func(self, mod: IRModule) -> IRModule:
        """Perform f_transform with its arguments
        Returns
        -------
        ret: Callable
           registered transformation function
        """
        return _ffi_api.ChoiceApplyTransformFunc(self, mod)

    def check_constr(self, mod: IRModule) -> bool:
        """Perform f_constr with its arguments
        Returns
        -------
        ret: bool
           Returns whether the IRModule satisfies the constraint or not
        """
        return _ffi_api.ChoiceCheckConstr(self, mod)

    def as_json(self) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object
        Returns
        -------
        json: JSON_TYPE
            The JSON-style object
        """
        return _ffi_api.ChoiceAsJSON(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Choice":
        """Create Choice from JSON obj

        Parameters
        ----------
        json_obj: JSON_TYPE
            Choice serialized with JSON

        Return
        ----------
        choice: Choice
            Deserialized choice
        """
        return _ffi_api.ChoiceFromJSON(json_obj)


@register_object("relax.tuning_api.Knob")
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
        @tvm.register_func("relax.tuning_api.test.f_transform")
        def apply(mod):
            return relax.tuning_api.FoldConstant()(mod)
        choices = {"apply": Choice("relax.tuning_api.test.f_transform"), "noapply": Choice()}
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

    def as_json(self) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object
        Returns
        -------
        json: JSON_TYPE
            The JSON-style object
        """
        return _ffi_api.KnobAsJSON(self)

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Knob":
        """Create Knob from JSON obj

        Parameters
        ----------
        json_obj: JSON_TYPE
            Knob serialized with JSON

        Return
        ----------
        knob: Knob
            Deserialized knob
        """
        return _ffi_api.KnobFromJSON(json_obj)

    def __str__(self) -> str:
        msg = f"{self.name} (# of choices: {len(self.choices)})\n"
        for name, choice in self.choices.items():
            msg += f"  - {name}: {choice}\n"
        return msg


@register_object("relax.tuning_api.Trace")
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

    def set_out_mod(self, mod: IRModule) -> None:
        """Set out_mod for the trace."""
        return _ffi_api.TraceSetOutMod(self, mod)

    def as_json(self, include_irmod: bool = True) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object.
        Parameters
        ----------
        include_irmod: bool
            Decides whether to serialize in_mod as well.

        Returns
        -------
        json: JSON_TYPE
            The JSON-style object.
        """
        obj = _ffi_api.TraceAsJSON(self, include_irmod)
        return _json_from_tvm(obj)

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Trace":
        """Create Trace from JSON obj.

        Parameters
        ----------
        json_obj: JSON_TYPE
            Trace serialized with JSON.

        Return
        ----------
        trace: Trace
            Deserialized trace.
        """
        return _ffi_api.TraceFromJSON(json_obj)

    def __str__(self) -> str:
        n = len(self.knobs)
        msg = f"Trace length: {n}\n"
        for idx in range(n):
            msg += f"[{idx+1}] {self.knobs[idx].name}: {self.decisions[idx]}\n"
        return msg


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
