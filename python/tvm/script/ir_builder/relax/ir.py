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

from tvm._ffi import register_object as _register_object
from tvm.ir import Attrs, Type
from tvm.relax import Expr, ShapeExpr, Var
from tvm.relax import Call, ExternFunc
from tvm.relax import op as _op
from tvm.runtime import Object
from tvm.tir import PrimExpr

from . import _ffi_api
from . import frame
from ..tir import var as tir_var


############################## Tensor Type ##############################
@_register_object("script.ir_builder.relax.TensorType")
class TensorType(Object):
    ...


def tensor_decl(
    shape: Optional[List[Union[PrimExpr, str]]],
    dtype: str,
    ndim: int = -1,
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


def ret_type(type: TensorType) -> None:
    return _ffi_api.RetType(type)  # pylint: disable=no-member # type: ignore


def func_return(type: Expr) -> Expr:
    return _ffi_api.FuncReturn(type)  # pylint: disable=no-member # type: ignore


############################# BindingBlock ##############################


def dataflow() -> frame.BlockFrame:
    return _ffi_api.Dataflow()  # pylint: disable=no-member # type: ignore


def binding_block() -> frame.BlockFrame:
    return _ffi_api.BindingBlock()  # pylint: disable=no-member # type: ignore


############################### Bindings ###############################


def emit(expr: Expr) -> Var:
    return _ffi_api.Emit(expr)  # type: ignore


def emit_match_shape(value: Expr, pattern: List[PrimExpr]) -> Var:
    return _ffi_api.EmitMatchShape(value, pattern)  # type: ignore


def emit_match_shape_without_var(value: Expr, pattern: List[PrimExpr]):
    return _ffi_api.EmitMatchShapeWithoutVar(value, pattern)  # type: ignore


################################# Ops #################################


def call_packed(
    func: str,
    *args,
    attrs: Optional[Attrs] = None,
    type_args: Optional[Union[TensorType, List[TensorType]]] = None,
):
    op = ExternFunc(func)
    if type_args is None:
        raise ValueError(f"R.call_packed is required to have type_args")
    if isinstance(type_args, TensorType):
        type_args = [type_args]
    elif isinstance(type_args, tuple):
        type_args = list(type_args)
    for i, arg in enumerate(type_args):
        if isinstance(arg, TensorType):
            type_args[i] = arg.type
        else:
            raise TypeError(
                "call_packed `type_args` is expected to be list of TensorType, "
                f"but got {type(arg)}"
            )

    return Call(op, args, attrs=attrs, type_args=type_args)


class MatchShapePair:
    value: Expr
    pattern: List[PrimExpr]

    def __init__(self, value: Expr, pattern: List[PrimExpr]) -> None:
        self.value = value
        self.pattern = pattern


def match_shape(value: Expr, pattern: List[PrimExpr]):
    return MatchShapePair(value, pattern)


call_tir = _op.call_tir
add = _op.add
multiply = _op.multiply


class builtin:
    @staticmethod
    def alloc_tensor(
        shape: Union[PrimExpr, List[PrimExpr]],
        dtype: str,
        runtime_device_index: int,
    ) -> Call:
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ShapeExpr(shape)
        return _op.builtin_alloc_tensor(shape, dtype, runtime_device_index)


__all__ = [
    "MatchShapePair",
    "TensorType",
    "add",
    "arg",
    "binding_block",
    "builtin",
    "call_packed",
    "call_tir",
    "dataflow",
    "emit",
    "emit_match_shape",
    "emit_match_shape_without_var",
    "func_attr",
    "func_name",
    "func_return",
    "function",
    "match_shape",
    "multiply",
    "ret_type",
    "tensor_decl",
]
