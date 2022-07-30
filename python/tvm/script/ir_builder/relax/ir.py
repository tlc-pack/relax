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
# pylint: disable=missing-docstring
"""IRBuilder for Relax"""

from typing import Dict, List, Optional, Union

from tvm.ir import Attrs, Type
from tvm.relax import Expr, Var
from tvm.relax import Call, ExternFunc
from tvm.relax import op as _op

from tvm.runtime import Object
from tvm.tir import PrimExpr

from . import _ffi_api
from . import frame
from ..tir import var as tir_var


############################## Tensor Type ##############################


def tensor_decl(
    shape: Optional[List[Union[PrimExpr, str]]],
    dtype: str,
    ndim: Optional[int] = None,
):
    if isinstance(shape, (tuple, list)):
        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, str):
                shape[i] = tir_var("int64", s)

    return _ffi_api.Tensor(shape, dtype, ndim)  # pylint: disable=no-member # type: ignore


############################### Function ################################


def function() -> frame.FunctionFrame:
    return _ffi_api.Function()  # pylint: disable=no-member # type: ignore


def arg(name: str, var: Var) -> Var:
    return _ffi_api.Arg(name, var)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> str:
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Object]) -> None:
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def ret_type(type: Type) -> Type:
    return _ffi_api.RetType(type)  # pylint: disable=no-member # type: ignore


def func_return(type: Type) -> Type:
    return _ffi_api.FuncReturn(type)  # pylint: disable=no-member # type: ignore


############################# BindingBlock ##############################


def dataflow() -> frame.BlockFrame:
    return _ffi_api.Dataflow()  # pylint: disable=no-member # type: ignore


def BindingBlock() -> frame.BlockFrame:
    return _ffi_api.BindingBlock()  # pylint: disable=no-member # type: ignore


############################### Bindings ###############################


def emit(expr: Expr) -> Var:
    return _ffi_api.Emit(expr)  # type: ignore


################################# Ops #################################


def call_packed(
    func: str,
    *args,
    attrs: Optional[Attrs] = None,
    type_args: Optional[Union[Type, List[Type]]] = None,
):

    op = ExternFunc(func)
    if type_args is None:
        raise ValueError(f"R.call_packed is required to have type_args")
    if not isinstance(type_args, list):
        type_args = [type_args]
    for i, arg in enumerate(type_args):
        if isinstance(arg, Var):
            type_args[i] = arg._checked_type_

    return Call(op, args, attrs=attrs, type_args=type_args)


call_tir = _op.call_tir
add = _op.add
multiply = _op.multiply


__all__ = [
    "BindingBlock",
    "add",
    "arg",
    "call_packed",
    "call_tir",
    "dataflow",
    "emit",
    "func_attr",
    "func_name",
    "func_return",
    "function",
    "multiply",
    "ret_type",
    "tensor_decl",
]
