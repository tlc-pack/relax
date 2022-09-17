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
# pylint: disable=missing-docstring,

import contextlib
from typing import Any, Tuple

from tvm import tir, relax
from tvm.script.ir_builder.relax.frame import BlockFrame

from ...ir_builder import relax as R
from ...ir_builder.base import IRBuilder
from .._core import Parser, dispatch, doc
from .entry import MatchShapePair


def bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    var_table = self.var_table.get()

    if isinstance(value, tir.Var):
        if value.name and var_name != value.name:
            self.report_error(
                node,
                "Cannot define TIR variables with different names. The LHS of binding should has "
                "the same name provided in RHS.",
            )
        if var_name in var_table:
            prev_value = var_table[var_name]
            if not isinstance(prev_value, tir.Var):
                self.report_error(
                    node,
                    "Cannot redefine a non-TIR-variable object to a TIR variable. Please define "
                    "the TIR variable with another name.",
                )
            if prev_value.dtype != value.dtype:
                self.report_error(
                    node,
                    "Expected the same dtype for TIR vars "
                    f"but got {value.dtype} vs {prev_value.dtype}",
                )
            return prev_value
        IRBuilder.name(var_name, value)
        return value

    if isinstance(value, relax.Expr):
        var = R.emit(value, var_name)
        # It's an internal check, so directly use assert here.
        assert var is not None
        return var
    elif isinstance(value, MatchShapePair):
        var = R.emit_match_shape(value.value, value.pattern, var_name)
        # It's an internal check, so directly use assert here.
        assert var is not None
        return var
    else:
        raise TypeError(f"Unsupported type {type(value)} in assignment")


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        with R.function():
            R.func_name(node.name)
            if node.returns is not None:
                R.func_ret_type(self.eval_expr(node.returns).type)
            with self.with_dispatch_token("relax"):
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="relax", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.FunctionDef) -> None:
    value = self.eval_expr(node.value)

    if isinstance(value, MatchShapePair):
        R.emit_match_shape(value.value, value.pattern, var_name=None)
    elif isinstance(value, tuple):
        # Currently `res` must be the return value of `R.output`. In order to make these variables
        # accessible to the bindings of following binding blocks, we should pop these variables into
        # the variable table of one level higher.
        for var_name in self.var_table.frames[-1].vars:
            if self.var_table.name2value[var_name][-1] in value:
                var = self.var_table.name2value[var_name][-1]
                # Pop up the variable to the variable table one level higher.
                if var_name in self.var_table.frames[-2].vars:
                    self.var_table.name2value[var_name][-2] = var
                else:
                    self.var_table.frames[-2].add(var_name)
                    self.var_table.name2value[var_name].append(var)
    elif value is not None:
        self.report_error(node, f"Unsupported Expr stmt type {value}.")


@dispatch.register(token="relax", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_type = self.visit_tvm_annotation(arg.annotation)
        param = R.arg(arg.arg, param_type)
        # Define the symbolic shape var
        if param_type.shape is not None:
            for shape_expr in param_type.shape:
                if isinstance(shape_expr, tir.Var):
                    self.var_table.add(shape_expr.name, shape_expr)

        self.var_table.add(arg.arg, param)


@dispatch.register(token="relax", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


@dispatch.register(token="relax", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    # Currently only `with R.dataflow()` is supported
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        if len(node.items) != 1:
            self.report_error(node, "Only one dataflow block is allowed")
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, BlockFrame):
                self.report_error(
                    item.context_expr, "Invalid context expression in the with-statement."
                )
            stack.enter_context(frame)
            if item.optional_vars is not None:
                self.report_error(
                    item.context_expr,
                    "Relax syntax doesn't allow binding expressions in `with` to variables",
                )

        # Since we don't know which variables are global variables during variable creation within
        # single round of visit, we adopt a two-round visit to deal with the construction of
        # dataflow block.
        # - In the first round, all binding variables are created as dataflow variables.
        # - At the end of the first round, by looking into the arguments of `R.output`, we know and
        # stores the names of the global variables.
        # - Then we clear the variable table, as a preparation step for the second round of visit.
        # - In the second round, we create variables according to their names, by checking whether
        # the name exists in the stored global variable names.

        # First round of visit
        self.visit(node.body)
        # Clear `var_table` in order to do a second round of visit
        for var in self.var_table.frames[-1].vars:
            self.var_table.name2value[var].pop()
        self.var_table.frames[-1].vars.clear()
        # Second round of visit
        self.visit(node.body)


@dispatch.register(token="relax", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value, allow_shadowing=True)


@dispatch.register(token="relax", type_name="Return")
def visit_return(self: Parser, node: doc.Assign) -> None:
    value = self.eval_expr(node.value)

    if isinstance(value, relax.Expr):
        R.func_ret_value(value)
    elif isinstance(value, Tuple):
        if all([isinstance(f, tir.PrimExpr) for f in value]):
            R.func_ret_value(relax.ShapeExpr(value))
        elif any([isinstance(f, tir.PrimExpr) for f in value]):
            self.report_error(
                node, "Return types, with mixed PrimExpr and Relax Expr, is not supported."
            )
        else:
            R.func_ret_value(relax.Tuple(value))
    else:
        self.report_error(node, f"Unsupported return value type {type(value)}.")
