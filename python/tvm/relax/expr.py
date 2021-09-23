# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import List, Optional, Union, Dict
import tvm._ffi
from ..ir import Node, Span, SourceName, BaseFunc
from ..runtime import String
from ..relay import Id, Tuple, TupleGetItem
from ..tir import PrimExpr
from . import _ffi_api
from .. import relay

Expr = relay.Expr
Type = relay.Type
GlobalVar = relay.GlobalVar
Call = relay.Call
If = relay.If
const = relay.const


@tvm._ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(Expr):
    values: List[PrimExpr]

    def __init__(self, values: List[PrimExpr], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values, span)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.values[index]

    def __len__(self):
        return len(self.values)


def make_shape(shape: List[PrimExpr]) -> ShapeExpr:
    if isinstance(shape, (list, tuple)):
        return ShapeExpr(shape)
    else:
        raise ValueError


@tvm._ffi.register_object("relax.expr.Var")
class Var(Expr):
    vid: Id
    type_annotation: Optional[Type]

    def __init__(
        self,
        name_hint: str,
        shape_annotation: Optional[Expr] = None,
        type_annotation: Optional[Type] = None,
        span: Span = None,
    ) -> None:
        if shape_annotation is not None:
            shape_annotation = make_shape(shape_annotation)
        self.__init_handle_by_constructor__(
            _ffi_api.Var, name_hint, shape_annotation, type_annotation, span
        )

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@tvm._ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    def __init__(
        self,
        name_hint: str,
        shape_annotation: Optional[Expr] = None,
        type_annotation: Optional[Type] = None,
        span: Span = None,
    ) -> None:
        if shape_annotation is not None:
            shape_annotation = make_shape(shape_annotation)
        self.__init_handle_by_constructor__(
            _ffi_api.DataflowVar, name_hint, shape_annotation, type_annotation, span
        )


@tvm._ffi.register_object("relax.expr.Binding")
class Binding(Node):
    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Binding, span)


@tvm._ffi.register_object("relax.expr.MatchShape")
class MatchShape(Binding):
    pattern: List[PrimExpr]
    value: Expr

    def __init__(self, pattern: List[PrimExpr], value: Expr, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MatchShape, pattern, value, span)


@tvm._ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    var: Var
    value: Expr

    def __init__(self, var: Var, value: Expr, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value, span)


@tvm._ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node):
    bindings: List[Binding]

    def __init__(self, bindings: List[Binding], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings, span)


@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    def __init__(self, bindings: List[Binding], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, span)


@tvm._ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(Expr):
    blocks: List[BindingBlock]
    body: Expr

    def __init__(self, blocks: List[BindingBlock], body: Expr, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, span)


@tvm._ffi.register_object("relax.expr.Function")
class Function(BaseFunc):
    name: Optional[GlobalVar]
    params: List[Var]
    body: Expr
    ret_type: Type

    def __init__(
        self,
        params: List[Var],
        body: Expr,
        ret_type: Type,
        name: Optional[GlobalVar] = None,
        span: Span = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Function, name, params, body, ret_type, span)


@tvm._ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc):
    global_symbol: String

    def __init__(self, global_symbol: String, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ExternFunc, global_symbol, span)


def extern(name, span: Span = None):
    return ExternFunc(name, span)
