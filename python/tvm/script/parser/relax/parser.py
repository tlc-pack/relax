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

from typing import Any

from tvm import tir

from ...ir_builder import relax as R
from ...ir_builder.base import name
from .._core import Parser, dispatch, doc


def bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    var = R.emit(value)
    name(var_name, var)
    return var


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        with R.function():
            R.func_name(node.name)
            if node.returns is not None:
                R.ret_type(self.eval_expr(node.returns).type)
            with self.with_dispatch_token("relax"):
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="relax", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        type = self.visit_tvm_annotation(arg.annotation)
        param = R.arg(arg.arg, type)
        # Define the symbolic shape var
        for shape_expr in type.shape:
            if isinstance(shape_expr, tir.Var):
                self.var_table.add(shape_expr.name, shape_expr)

        self.var_table.add(arg.arg, param)


@dispatch.register(token="relax", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


@dispatch.register(token="relax", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    value_table = self.var_table.get()
    if (
        isinstance(value_table.get(lhs.id, None), tir.Var)
        and isinstance(rhs, tir.Var)
        and rhs.name == lhs.id
    ):
        # Support redefine symbolic shape
        # e.g. m = T.var("int64", "m")
        pass
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="relax", type_name="Return")
def visit_return(self: Parser, node: doc.Assign) -> None:
    value = self.eval_expr(node.value)
    R.func_return(value)
