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
from ..ir.base import Node, Span, SourceName
from ..relay.base import Id
from ..tir import PrimExpr
from . import _ffi_api
from .. import relay

GlobalVar = relay.GlobalVar
Expr = relay.Expr
Type = relay.Type
const = relay.const

@tvm._ffi.register_object("relax.expr.Var")
class Var(Expr):
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

    def __init__(self, pattern: List[PrimExpr], value: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.MatchShape, pattern, value, span) # type: ignore

@tvm._ffi.register_object("relax.expr.VarBinding")
class VarBinding(Node):
    var: Var
    val: Expr

    def __init__(self, var: Var, val: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Binding, var, val, span) # type: ignore


@tvm._ffi.register_object("relax.expr.BindingBlock")
class BasicBlock(Node):
    bindings: List[Binding]

    def __init__(self, bindings: List[Binding], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BasicBlock, bindings, span) # type: ignore

@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataFlowBlock(Node):
    bindings: List[Binding]

    def __init__(self, bindings: List[Binding], span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, span) # type: ignore

@tvm._ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(Expr):
    blocks: List[BasicBlock]
    body: Expr

    def __init__(self, blocks: List[BasicBlock], body: Expr, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, span) # type: ignore


@tvm._ffi.register_object("relax.expr.Function")
class Function(Expr):
    name: Optional[GlobalVar]
    params: List[Var]
    body: Expr
    ret_type: Type

    def __init__(self, name: Optional[GlobalVar], params: List[Var], body: Expr, ret_type: Type, span: Span) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Function,  name,  params,  body,  ret_type,  span) # type: ignore
