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
"""The base parser for ir module"""
from typing import Optional, Tuple

from tvm.ir import PrimExpr, PrimType, RelayExpr, Type
from ...ir_builder import ir as I
from .._core import Parser, dispatch, doc


def eval_func_type_shape(
    self: Parser, node: doc.FunctionDef
) -> Tuple[Optional[Type], Optional[RelayExpr]]:
    """evaluate function type and shape.
    Parameters
    ----------
    self : Parser
        The visiting parser.
    node : doc.FunctionDef
        The doc FunctionDef node.
    """
    token = self.get_dispatch_token(node)
    with self.with_dispatch_token(token):
        result = self.visit_tvm_annotation(node.returns)
    if result is None:
        return None, None
    elif isinstance(result, tuple) and len(result) == 2:
        # relax dialect
        return result
    elif isinstance(result, PrimExpr):
        # tir dialect
        return PrimType(result.dtype), None
    else:
        raise TypeError(f"Unsupported annotation type: {result}")


@dispatch.register(token="ir", type_name="ClassDef")
def _visit_class_def(self: Parser, node: doc.ClassDef) -> None:
    """The class definition visiting method for ir module.
    Parameters
    ----------
    self : Parser
        The visiting parser.
    node : doc.ClassDef
        The doc AST class definition node.
    """

    with self.var_table.with_frame():
        with I.ir_module():
            for stmt in node.body:
                if isinstance(stmt, doc.FunctionDef):
                    self.visit_tvm_declare_function(stmt)
            with self.with_dispatch_token("ir"):
                self.visit_body(node.body)


@dispatch.register(token="ir", type_name="Assign")
def _visit_assign(_self: Parser, _node: doc.Assign) -> None:
    """The assign visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST assign node.
    """


@dispatch.register(token="ir", type_name="Expr")
def _visit_expr(_self: Parser, _node: doc.Expr) -> None:
    """The expression visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST expression node.
    """


@dispatch.register(token="default", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> None:
    global_var = I.decl_function(node.name)
    self.var_table.add(node.name, global_var)
