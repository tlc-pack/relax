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

import numbers
from typing import Any, Optional, Tuple, Union

from tvm import relax, tir
from tvm.ir import Type
from tvm.relax import StructInfo, Expr
from tvm.relax.utils import convert_to_expr
from tvm.script.ir_builder.relax.frame import BlockFrame

from ...ir_builder import ir as I
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

    if isinstance(value, tuple):
        value = convert_to_expr(value)
    if isinstance(value, numbers.Number):
        value = R.const(value)
    if isinstance(value, relax.Expr):
        var = R.emit(value)
        # It's an internal check, so directly use assert here.
        assert var is not None
        IRBuilder.name(var_name, var)
        return var
    elif isinstance(value, MatchShapePair):
        var = R.emit_match_shape(value.value, value.pattern, emit_var=True)
        # It's an internal check, so directly use assert here.
        assert var is not None
        IRBuilder.name(var_name, var)
        return var
    else:
        raise TypeError(f"Unsupported type {type(value)} in assignment")


def eval_shape_annotation(
    self: Parser, node: Union[doc.Expression, doc.expr], shape: relax.Expr
) -> Any:
    if shape is None:
        return None
    if isinstance(shape, relax.RuntimeDepShape):
        return shape
    elif isinstance(shape, relax.ShapeExpr):
        shape = list(shape.values)
        for i, expr in enumerate(shape):
            # Define the symbolic shape var
            if isinstance(expr, tir.Var):
                name = expr.name
                if name in self.var_table.get():
                    shape[i] = self.var_table.get()[name]
                else:
                    self.var_table.add(name, shape[i])
        return relax.ShapeExpr(shape)
    elif isinstance(shape, relax.Tuple):
        shapes = [eval_shape_annotation(self, node, s) for s in shape.fields]
        if any([s is None for s in shapes]):
            return None
        return relax.Tuple(shapes)
    else:
        self.report_error(node, f"Unsupported shape {type(shape)}")
        return None


# pylint: disable=inconsistent-return-statements
def eval_type_annotation(
    self: Parser, node: Union[doc.Expression, doc.expr]
) -> Tuple[Type, Optional[Expr], StructInfo]:
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    if isinstance(annotation, StructInfo):
        var_table = {k: v for k, v in self.var_table.get().items() if isinstance(v, tir.Var)}
        annotation, undefined_vars = R.RewriteSymbolicShape(annotation, var_table)
        for var in undefined_vars:
            self.var_table.add(var.name, var)
        return annotation
    else:
        self.report_error(node, f"Unsupported type annotation {annotation}")


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        with R.function():
            R.func_name(node.name)
            if node.returns is not None:
                ann_sinfo = eval_type_annotation(self, node.returns)
                R.func_ret_struct_info(ann_sinfo)

            with self.with_dispatch_token("relax"):
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="relax", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> None:
    if node.returns is None:
        ret_sinfo = relax.TupleStructInfo([])
    else:
        ret_sinfo = eval_type_annotation(self, node.returns)
    params = []
    params_sinfo = []
    for arg in node.args.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_sinfo = self.visit_tvm_annotation(arg.annotation)
        params_sinfo.append(param_sinfo)
        params.append(relax.Var(arg.arg, param_sinfo))

    func_signature = relax.Function.create_empty(params, ret_sinfo)
    global_var = I.decl_function(node.name, func_signature)
    self.var_table.add(node.name, global_var)


@dispatch.register(token="relax", type_name="pre_token_switch")
def pre_token_switch(self: Parser, node: doc.Expr) -> None:  # pylint: disable=unused-argument
    ir_builder = IRBuilder()
    ir_builder.__enter__()


@dispatch.register(token="relax", type_name="post_token_switch")
def post_token_switch(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder.current()
    result = ir_builder.get()
    ir_builder.__exit__(None, None, None)
    var = R.emit(result)
    IRBuilder.name(node.name, var)
    self.var_table.add(node.name, var, allow_shadowing=False)


@dispatch.register(token="relax", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    value = self.eval_expr(node.value)
    if isinstance(value, MatchShapePair):
        R.emit_match_shape(value.value, value.pattern, emit_var=False)
    elif value is not None:
        self.report_error(node, f"Unsupported Expr stmt type {value}.")


@dispatch.register(token="relax", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_sinfo = self.visit_tvm_annotation(arg.annotation)
        param = R.arg(arg.arg, param_sinfo)

        self.var_table.add(arg.arg, param)


@dispatch.register(token="relax", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    return eval_type_annotation(self, node)


@dispatch.register(token="relax", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    # Currently only `with R.dataflow()` is supported
    if len(node.items) != 1:
        self.report_error(node, "Only one item is allowed.")
    item = node.items[0]
    if item.optional_vars is not None:
        self.report_error(
            item.context_expr,
            "Relax syntax doesn't allow binding expressions in `with` to variables",
        )
    frame = self.eval_expr(item.context_expr)
    with self.var_table.with_frame():
        with frame:
            self.visit(node.body)
    if isinstance(frame, BlockFrame) and frame.is_dataflow:
        output_vars = frame.output_vars
        for var in output_vars:
            self.var_table.add(var.name_hint, var, allow_shadowing=True)


@dispatch.register(token="relax", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=bind_assign_value,
        allow_shadowing=True,
    )


@dispatch.register(token="relax", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_sinfo = self.visit_tvm_annotation(node.annotation)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=bind_assign_value,
        allow_shadowing=True,
    )
    var = self.var_table.get().get(lhs.id)
    assert isinstance(var, relax.Var)
    R.ir.annotate_struct_info(var, ann_sinfo)


@dispatch.register(token="relax", type_name="Return")
def visit_return(self: Parser, node: doc.Assign) -> None:
    value = self.eval_expr(node.value)
    value = convert_to_expr(value)
    R.func_ret_value(value)


@dispatch.register(token="relax", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    if node.orelse is None:
        raise ValueError("Else statements are required for relax dialect.")
    with R.If(self.eval_expr(node.test)) as if_frame:
        with self.var_table.with_frame():
            with R.Then():
                self.visit_body(node.body)
        with self.var_table.with_frame():
            with R.Else():
                self.visit_body(node.orelse)
    self.var_table.add(if_frame.var_name, if_frame.var, allow_shadowing=True)
