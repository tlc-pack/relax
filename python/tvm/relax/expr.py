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
const = relay.const


@tvm._ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(Expr):
    values: List[PrimExpr]

    def __init__(self, values: List[PrimExpr]) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values)


@tvm._ffi.register_object("relax.expr.Var")
class Var(Expr):
    id: Id
    type_annotation: Optional[Type]

    def __init__(self, name_hint: str,
                 shape_annotation: Optional[List[Type]] = None,
                 type_annotation: Optional[Type] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Var, name_hint,
                                            shape_annotation,
                                            type_annotation)

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@tvm._ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    pass


@tvm._ffi.register_object("relax.expr.Binding")
class Binding(Node):
    pass


@tvm._ffi.register_object("relax.expr.MatchShape")
class MatchShape(Binding):
    pattern: List[PrimExpr]
    value: Expr

    def __init__(self, pattern: List[PrimExpr], value: Expr) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MatchShape, pattern, value)


@tvm._ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    var: Var
    value: Expr

    def __init__(self, var: Var, value: Expr) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value)


@tvm._ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node):
    bindings: List[Binding]

    def __init__(self, bindings: List[Binding]) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings)


@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    pass


@tvm._ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(Expr):
    blocks: List[BindingBlock]
    body: Expr

    def __init__(self, blocks: List[BindingBlock], body: Expr) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body)


@tvm._ffi.register_object("relax.expr.Function")
class Function(BaseFunc):
    name: Optional[GlobalVar]
    params: List[Var]
    body: Expr
    ret_type: Type

    def __init__(self, params: List[Var], body: Expr,
                 ret_type: Type, name: Optional[GlobalVar] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Function, name, params,
                                            body, ret_type)


@tvm._ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc):
    global_symbol: String

    def __init__(self, global_symbol: String) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ExternFunc, global_symbol)

def extern(name):
    return ExternFunc(name)
