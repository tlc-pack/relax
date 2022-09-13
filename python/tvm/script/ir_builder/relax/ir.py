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

from typing import Dict, List, Optional, Union

from tvm._ffi import register_object as _register_object
from tvm.ir import Type
from tvm.relax import Expr, Var
from tvm.relax.op import call_tir
from tvm.runtime import Object
from tvm.tir import PrimExpr

from . import _ffi_api
from . import frame

############################### Operators ###############################
from tvm.relax.op import shape_of, make_closure, invoke_closure
from tvm.relax.op import add, multiply, unique
from tvm.relax.op import builtin


############################## Tensor Type ##############################


@_register_object("script.ir_builder.relax.TensorType")
class TensorType(Object):
    """A temporary Tensor type for `R.Tensor` in ir_builder."""


def tensor(
    shape: Optional[List[PrimExpr]] = None,
    dtype: Optional[str] = None,
    ndim: int = -1,
):
    """Helper function for `R.Tensor` in parser

    Parameters
    ----------
    shape: Optional[List[PrimExpr]]
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


def binding_block() -> frame.BlockFrame:
    """Start a binding block frame.

    Returns
    -------
    frame: frame.BlockFrame
        The created ir_builder Block frame.
    """
    return _ffi_api.BindingBlock()  # pylint: disable=no-member # type: ignore


def dataflow() -> frame.BlockFrame:
    """Start a dataflow binding block frame.

    Returns
    -------
    frame: frame.BlockFrame
        The created ir_builder Block frame.
    """
    return _ffi_api.Dataflow()  # pylint: disable=no-member # type: ignore


############################### Bindings ###############################


def emit(value: Expr) -> Var:
    """Emit a binding to the last binding block frame.

    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.

    """
    return _ffi_api.Emit(value)  # type: ignore


__all__ = [
    "TensorType",
    "add",
    "arg",
    "binding_block",
    "builtin",
    "call_tir",
    "dataflow",
    "emit",
    "func_attr",
    "func_name",
    "func_ret_type",
    "func_ret_value",
    "function",
    "invoke_closure",
    "make_closure",
    "multiply",
    "unique",
    "shape_of",
    "tensor",
]
