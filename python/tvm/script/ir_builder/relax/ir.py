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
# pylint: disable=redefined-builtin, wrong-import-order
"""IRBuilder for Relax dialect"""

from typing import Dict, List, Optional, Tuple, Union

from tvm._ffi import register_object as _register_object
from tvm.ir import Type
from tvm.relax import Expr, Var
from tvm.relax.op import call_tir
from tvm.runtime import Object
from tvm.tir import PrimExpr

from . import _ffi_api
from . import frame
from ..tir import var as _tir_var

############################### Operators ###############################
from tvm.relax.op import shape_of, make_closure, invoke_closure
from tvm.relax.op import add, multiply, unique
from tvm.relax.op import builtin


############################## Tensor Type ##############################


@_register_object("script.ir_builder.relax.TensorType")
class TensorType(Object):
    """A temporary Tensor type for `R.Tensor` in ir_builder."""


def tensor(
    shape: Optional[List[Union[PrimExpr, str]]] = None,
    dtype: Optional[str] = None,
    ndim: int = -1,
):
    """Helper function for `R.Tensor` in parser

    Parameters
    ----------
    shape: Optional[List[Union[PrimExpr, str]]]
        The shape of the tensor. It's runtime dependent if `shape` is None.

    dtype: Optional[str]
        The element data type of the tensor. It's runtime dependent if `dtype` is None.

    ndim: int
        The number of dimensions of the tensor. It's runtime dependent if `ndim` is -1.

    Returns
    -------
    tensor_type: TensorType
        The TensorType that is only used in ir_builder.
    """
    if isinstance(shape, (tuple, list)):
        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, str):
                shape[i] = _tir_var("int64", s)

    return _ffi_api.Tensor(shape, dtype, ndim)  # pylint: disable=no-member # type: ignore


############################### Function ################################


def function() -> frame.FunctionFrame:
    """Start a function frame.

    Returns
    -------
    frame: FunctionFrame
        The constructed function frame.
    """
    return _ffi_api.Function()  # pylint: disable=no-member # type: ignore


def arg(name: str, type: TensorType) -> Var:
    """Add a parameter to the last function frame.

    Parameters
    ----------
    name: str
        The name of the parameter.

    type: TensorType
        The type and the shape of the parameter.

    Returns
    -------
    var: Var
        The created function parameter var.
    """
    return _ffi_api.Arg(name, type)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> None:
    """Specify the name of the last function frame.

    Parameters
    ----------
    name: str
        The function name.
    """
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Object]) -> None:
    """Specify the attrs of the last function frame.

    Parameters
    ----------
    attrs: Dict[str, Object]
        The function attrs.
    """
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret_type(ret_type: Union[TensorType, Type]) -> None:
    """Specify the return type of the last function frame.

    Parameters
    ----------
    ret_type: Union[TensorType, Type]
        The function return type.
    """
    if isinstance(ret_type, TensorType):
        ret_type = ret_type.type
    return _ffi_api.FuncRetType(ret_type)  # pylint: disable=no-member # type: ignore


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


def output(*vars: Tuple[Var]) -> Tuple[Var]:
    """Expose the dataflow block output variables as global ones.

    Parameters
    ----------
    vars: Tuple[Var]
        The output variables of a dataflow block.

    Returns
    -------
    vars: Tuple[Var]
        The output variables of a dataflow block. Return the input variables to parser side for
        followup process
    """
    _ffi_api.DataflowBlockOutput(vars)  # pylint: disable=no-member # type: ignore
    return vars


############################### Bindings ###############################


def emit(value: Expr, var_name: str) -> Var:
    """Emit a binding to the last binding block frame.

    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.
    var_name: str
        The variable name of the emitted binding.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.

    """
    return _ffi_api.Emit(value, var_name)  # pylint: disable=no-member # type: ignore


def emit_match_shape(
    value: Expr, pattern: List[PrimExpr], var_name: Optional[str]
) -> Optional[Var]:
    """Emit a match_shape binding to the last binding block frame.

    Parameters
    ----------
    value: Expr
        The value of the MatchShape to be emitted.
    pattern: List[PrimExpr]
        The pattern of the MatchShape to be emitted.
    var_name: Optional[str]
        The variable name of the emitted binding. `None` if we don't intend to emit the binding
        variable

    Returns
    -------
    var: Optional[Var]
        The emitted var if `var_name` is not `None`. Otherwise, return `None`.
    """
    return _ffi_api.EmitMatchShape(value, pattern, var_name)  # type: ignore


############################### Importer ###############################

__all__ = [
    "TensorType",
    "add",
    "arg",
    "builtin",
    "call_tir",
    "dataflow",
    "emit",
    "emit_match_shape",
    "func_attr",
    "func_name",
    "func_ret_type",
    "func_ret_value",
    "function",
    "invoke_closure",
    "make_closure",
    "multiply",
    "output",
    "unique",
    "shape_of",
    "tensor",
]
