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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from tvm import relax, tir
from tvm.ir import Type
from tvm.script.ir_builder.relax.frame import BlockFrame

from ...ir_builder import relax as R
from ...ir_builder.base import IRBuilder
from .._core import Parser, dispatch, doc, parse
from .entry import MatchShapePair, Tensor, TensorType


class VarDefLoc:
    def __init__(self, name: str, line: int, col: int):
        self.name = name
        self.line = line
        self.col = col

    def __str__(self):
        return f"{self.name}@{self.line}:{self.col}"

    def __repr__(self):
        return f"{self.name}@{self.line}:{self.col}"


def collect_var_definitions(stmts: List[doc.stmt]) -> Dict[str, List[VarDefLoc]]:
    class Collector(doc.NodeVisitor):
        results: Dict[str, List[VarDefLoc]]

        def __init__(self):
            self.results = defaultdict(list)

        def visit_Name(self, node: doc.Name):  # pylint: disable=invalid-name
            assert isinstance(node.ctx, doc.Store)
            assert node.id
            assert node.lineno is not None
            assert node.col_offset is not None
            self.results[node.id].append(
                VarDefLoc(
                    node.id,
                    node.lineno,
                    node.col_offset,
                )
            )

    collector = Collector()
    for stmt in stmts:
        if isinstance(stmt, doc.Assign):
            assert len(stmt.targets) == 1
            collector.visit(stmt.targets[0])
        elif isinstance(stmt, doc.AugAssign):
            collector.visit(stmt.target)

    return collector.results


def bind_value_with_dataflow_var_names(
    dataflow_var_names: List[str], var_def_table: Optional[Dict[str, List[VarDefLoc]]]
):
    def bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
        var_table = self.var_table.get()

        if isinstance(value, tir.Var):
            if value.name and var_name != value.name:
                self.report_error(
                    node,
                    "Cannot define TIR variables with different names. The LHS of binding should "
                    "has the same name provided in RHS.",
                )
            if var_name in var_table:
                prev_value = var_table[var_name]
                if not isinstance(prev_value, tir.Var):
                    self.report_error(
                        node,
                        "Cannot redefine a non-TIR-variable object to a TIR variable. Please "
                        "define the TIR variable with another name.",
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

        is_dataflow_var = False
        if var_def_table is not None and (
            var_name not in dataflow_var_names or node.lineno != var_def_table[var_name][-1].line
        ):
            is_dataflow_var = True

        if isinstance(value, relax.Expr):
            var = R.emit(value, is_dataflow_var)
            # It's an internal check, so directly use assert here.
            assert var is not None
            IRBuilder.name(var_name, var)
            return var
        elif isinstance(value, MatchShapePair):
            var = R.emit_match_shape(
                value.value, value.pattern, emit_var=True, is_dataflow_var=is_dataflow_var
            )
            # It's an internal check, so directly use assert here.
            assert var is not None
            IRBuilder.name(var_name, var)
            return var
        else:
            raise TypeError(f"Unsupported type {type(value)} in assignment")

    return bind_assign_value


def eval_type_annotation(self: Parser, node: Union[doc.Expression, doc.expr]) -> Any:
    type_annotation = self.eval_expr(node)
    if callable(type_annotation):
        type_annotation = Tensor()
    if isinstance(type_annotation, TensorType):
        shape = type_annotation.shape
        if shape is None:
            return type_annotation.type, None
        shape = list(shape.values)
        var_table = self.var_table.get()
        for i, expr in enumerate(shape):
            # Define the symbolic shape var
            if isinstance(expr, tir.Var):
                name = expr.name
                if name in var_table:
                    shape[i] = var_table[name]
                else:
                    self.var_table.add(name, shape[i])
        return type_annotation.type, relax.ShapeExpr(shape)
    else:
        if not isinstance(type_annotation, Type):
            self.report_error(node, f"Unsupported type annotation {type(type_annotation)}")
        return type_annotation, None


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        with R.function():
            R.func_name(node.name)
            if node.returns is not None:
                ann_type, _ = eval_type_annotation(self, node.returns)
                R.func_ret_type(ann_type)
            with self.with_dispatch_token("relax"):
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="relax", type_name="pre_token_switch")
def pre_token_switch(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder()
    ir_builder.__enter__()


@dispatch.register(token="relax", type_name="post_token_switch")
def pre_token_switch(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder.current()
    result = ir_builder.get()
    ir_builder.__exit__(None, None, None)
    var = R.emit(result, is_dataflow_var=False)
    IRBuilder.name(node.name, var)
    self.var_table.add(node.name, var, allow_shadowing=False)


@dispatch.register(token="relax", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    value = self.eval_expr(node.value)
    if isinstance(value, MatchShapePair):
        R.emit_match_shape(value.value, value.pattern, emit_var=False, is_dataflow_var=False)
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
        param_type, param_shape = self.visit_tvm_annotation(arg.annotation)
        param = R.arg(arg.arg, param_type, param_shape)

        self.var_table.add(arg.arg, param)


@dispatch.register(token="relax", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    return eval_type_annotation(self, node)


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

        assert isinstance(node.body, list)
        var_def_table = collect_var_definitions(node.body)

        if (
            not isinstance(node.body[-1], doc.Expr)
            or not isinstance(node.body[-1].value, doc.Call)
            or node.body[-1].value.func.attr != "output"
        ):
            self.report_error(
                node.body[-1],
                "Relax dataflow blocks must have output. However, the last statement inside a "
                "dataflow block is not `R.output`. Please use `R.output` to specify the output of "
                "the dataflow block.",
            )

        dataflow_var_names = []
        for arg in node.body[-1].value.args:
            if not isinstance(arg, doc.Name):
                self.report_error(
                    arg,
                    "The output of Relax dataflow blocks must be all variables. However, one of "
                    "the dataflow block output is not a variable. Please make sure all output are "
                    "variables.",
                )
            dataflow_var_names.append(arg.id)

        for i in range(len(node.body) - 1):
            if not isinstance(node.body[i], doc.Assign):
                self.report_error(
                    node.body[i],
                    "One non-assign statement appears unexpectedly inside a dataflow block. Only "
                    "the last statement inside a dataflow block is an Expr. Please make sure this "
                    "statement appears at a correct position.",
                )
            if len(node.body[i].targets) != 1:
                self.report_error(
                    node.body[i], "Consequential assignments like 'a = b = c' are not supported."
                )
            lhs = node.body[i].targets[0]
            rhs = self.eval_expr(node.body[i].value)
            self.eval_assign(
                target=lhs,
                source=rhs,
                bind_value=bind_value_with_dataflow_var_names(dataflow_var_names, var_def_table),
                allow_shadowing=True,
            )

        self.visit(node.body[-1])


@dispatch.register(token="relax", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=bind_value_with_dataflow_var_names(dataflow_var_names=[], var_def_table=None),
        allow_shadowing=True,
    )


@dispatch.register(token="relax", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_type, ann_shape = self.visit_tvm_annotation(node.annotation)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=bind_value_with_dataflow_var_names(dataflow_var_names=[], var_def_table=None),
        allow_shadowing=True,
    )
    var = self.var_table.get().get(lhs.id)
    assert isinstance(var, relax.Var)
    R.ir.annotate_type_shape(var, ann_type, ann_shape)


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
