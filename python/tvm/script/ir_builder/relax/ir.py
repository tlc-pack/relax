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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""IRBuilder for Relax dialect"""

import functools
import inspect
from typing import Dict, List, Optional, Tuple, Union

import tvm
from tvm.ir import Type
from tvm import relax
from tvm.relax import Call, Expr, ExternFunc, TupleGetItem, Var, const
from tvm.relax.struct_info import StructInfo
from tvm.relax.analysis import get_static_type

############################### Operators ###############################
from tvm.relax.op import (
    add,
    assert_op,
    builtin,
    null_value,
    call_tir,
    call_builtin,
    cos,
    divide,
    equal,
    ewise_fma,
    floor_divide,
    full,
    full_like,
    greater,
    greater_equal,
    image,
    invoke_closure,
    less,
    less_equal,
    log,
    make_closure,
    matmul,
    max,
    mean,
    memory,
    min,
    multiply,
    negative,
    nn,
    not_equal,
    ones,
    ones_like,
    print,
    prod,
    shape_of,
    sigmoid,
    sin,
    sqrt,
    std,
    subtract,
    sum,
    tanh,
    tril,
    triu,
    unique,
    variance,
    where,
    zeros,
    zeros_like,
)
from tvm.relax.utils import convert_to_expr
from tvm.runtime import Object as tvm_Object, ObjectGeneric

from . import _ffi_api, frame

##################### Python Native Function Alias ######################

py_print = print
py_tuple = tuple


############################### Function ################################


def function() -> frame.FunctionFrame:
    """Start a function frame.
    Returns
    -------
    frame: FunctionFrame
        The constructed function frame.
    """
    return _ffi_api.Function()  # pylint: disable=no-member # type: ignore


def arg(name: str, struct_info: StructInfo) -> Var:
    """Add a parameter to the last function frame.
    Parameters
    ----------
    name: str
        The name of the parameter.
    struct_info: StructInfo
        The Struct Info of the parameter

    Returns
    -------
    var: Var
        The created function parameter var.
    """

    return _ffi_api.Arg(name, struct_info)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> None:
    """Specify the name of the last function frame.
    Parameters
    ----------
    name: str
        The function name.
    """
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, tvm_Object]) -> None:
    """Specify the attrs of the last function frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The function attrs.
    """
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret_struct_info(ret_sinfo: StructInfo) -> None:
    """Specify the return struct info of the last function frame.
    Parameters
    ----------
    ret_type: StructInfo
        The function return struct info.
    """
    return _ffi_api.FuncRetStructInfo(ret_sinfo)  # pylint: disable=no-member # type: ignore


def func_ret_value(value: Expr) -> None:
    """Specify the return value of the last function frame.
    Parameters
    ----------
    value: Expr
        The function return value.
    """
    return _ffi_api.FuncRetValue(value)  # pylint: disable=no-member # type: ignore


############################# BindingBlock ##############################


def dataflow() -> frame.BlockFrame:
    """Start a dataflow binding block frame.
    Returns
    -------
    frame: frame.BlockFrame
        The created ir_builder Block frame.
    """
    return _ffi_api.Dataflow()  # pylint: disable=no-member # type: ignore


def output(*vars: Tuple[Var]) -> None:
    """Expose the dataflow block output variables as global ones.
    Parameters
    ----------
    vars: Tuple[Var]
        The output variables of a dataflow block.
    """
    return _ffi_api.DataflowBlockOutput(vars)  # pylint: disable=no-member # type: ignore


################################## Ops #################################


def call_packed(
    func: str,
    *args: List[Expr],
    type_args: Optional[Union[StructInfo, List[StructInfo]]] = None,
    **kwargs: Dict[str, Expr],
) -> Call:
    """Create a relax Call, which calls a packed function.
    Parameters
    ----------
    func: str
        The name of extern function.
    args : List[Expr]
        The arguments.
    type_args: Optional[Union[StructInfo, List[StructInfo]]]
        List of Types
    kwargs: Dict[str, Expr]
        The keyword arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    op = ExternFunc(func)
    args = [convert_to_expr(arg) for arg in args]
    if type_args is None:
        raise ValueError("R.call_packed is required to have type_args")
    if isinstance(type_args, py_tuple):
        type_args = list(type_args)
    elif not isinstance(type_args, list):
        type_args = [type_args]
    for i, argument in enumerate(type_args):
        if callable(argument):
            argument = argument()
        # Convert possible StructInfoProxy to StructInfo
        if isinstance(argument, ObjectGeneric):
            argument = argument.asobject()
        if isinstance(argument, StructInfo):
            type_args[i] = get_static_type(argument)
        elif isinstance(argument, Type):
            type_args[i] = argument
        else:
            raise TypeError(
                "call_packed `type_args` is expected to be list of StructInfo/Type, "
                f"but got {type(arg)}"
            )

    is_default = False
    if "attrs_type_key" in kwargs:
        attrs_type_key = kwargs["attrs_type_key"]
        kwargs.pop("attrs_type_key")
    else:
        attrs_type_key = "DictAttrs"
        is_default = True
    attrs = None
    if kwargs or not is_default:
        attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)

    return Call(op, args, attrs=attrs, type_args=type_args)


def _tensor_type_wrapper(func):
    """A wrapper to convert StructInfo to relax.DynTensorType"""

    def _convert_tensor_type(args):
        if isinstance(args, (list, py_tuple)):
            new_args = [_convert_tensor_type(x) for x in args]
            return type(args)(new_args)
        if isinstance(args, dict):
            return {_convert_tensor_type(k): _convert_tensor_type(v) for k, v in args.items()}
        if inspect.isfunction(args):
            args = args()
        if isinstance(args, ObjectGeneric):
            args = args.asobject()
        return get_static_type(args) if isinstance(args, StructInfo) else args

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*_convert_tensor_type(args), **_convert_tensor_type(kwargs))

    return wrapped  # type: ignore


invoke_closure = _tensor_type_wrapper(invoke_closure)  # pylint: disable=invalid-name

call_builtin = _tensor_type_wrapper(call_builtin)  # pylint: disable=invalid-name

############################### Bindings ###############################


def emit(value: Expr, annotate_struct_info: Optional[StructInfo] = None) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.

    annotate_struct_info: Optional[StructInfo]
        The optional struct info annotation for the emitted value.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.Emit(value, annotate_struct_info)  # pylint: disable=no-member # type: ignore


def emit_match_cast(value: Expr, struct_info: StructInfo) -> Var:
    """Emit a match_cast binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The value of the MatchCast to be emitted.
    struct_info: StructInfo
        The struct_info of the MatchCast to be emitted.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitMatchCast(value, struct_info)  # type: ignore


############################# If Then Else #############################


def If(condition: Expr) -> frame.IfFrame:  # pylint: disable=invalid-name
    """Create an if frame.
    Parameters
    ----------
    condition : Expr
        The condition of if statement, executes the true branch if the condition is true,
        otherwise jump into the false branch.
    Returns
    -------
    res : frame.IfFrame
        The result IfFrame.
    """
    return _ffi_api.If(condition)  # pylint: disable=no-member # type: ignore


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    """Create a then frame.
    Returns
    -------
    res : frame.ThenFrame
        The result ThenFrame.
    """
    return _ffi_api.Then()  # pylint: disable=no-member # type: ignore


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    """Create an else frame.
    Returns
    -------
    res : frame.ElseFrame
        The result ElseFrame.
    """
    return _ffi_api.Else()  # pylint: disable=no-member # type: ignore


############################### R.tuple ################################


def tuple(*fields: List[Expr]) -> Expr:
    """Create a tuple expression.
    Parameters
    ----------
    fields : List[Expr]
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    if len(fields) == 0:
        fields = []

    return relax.Tuple(fields)  # pylint: disable=no-member # type: ignore


############################### Importer ###############################

__all__ = [
    "Else",
    "If",
    "Then",
    "TupleGetItem",
    "add",
    "arg",
    "assert_op",
    "builtin",
    "call_packed",
    "call_tir",
    "call_builtin",
    "cos",
    "null_value",
    "const",
    "dataflow",
    "divide",
    "emit",
    "emit_match_cast",
    "equal",
    "ewise_fma",
    "floor_divide",
    "full",
    "full_like",
    "func_attr",
    "func_name",
    "func_ret_struct_info",
    "func_ret_value",
    "function",
    "greater",
    "greater_equal",
    "image",
    "invoke_closure",
    "less",
    "less_equal",
    "log",
    "make_closure",
    "matmul",
    "max",
    "mean",
    "memory",
    "min",
    "multiply",
    "negative",
    "nn",
    "not_equal",
    "ones",
    "ones_like",
    "output",
    "print",
    "prod",
    "shape_of",
    "sigmoid",
    "sin",
    "sqrt",
    "std",
    "subtract",
    "sum",
    "tanh",
    "tril",
    "triu",
    "tuple",
    "unique",
    "variance",
    "where",
    "zeros",
    "zeros_like",
]
