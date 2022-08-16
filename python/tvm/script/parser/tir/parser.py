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
import contextlib
from functools import partial
from typing import Any

from tvm.ir import PrimType
from tvm.tir import Buffer, IterVar, PrimExpr, Var

from ...ir_builder import tir as T
from ...ir_builder.base import IRBuilderFrame as Frame
from ...ir_builder.base import name
from .._core import Parser, dispatch, doc


def bind_with_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, (Buffer, Var)):
        name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in with statement")
        raise NotImplementedError


def bind_for_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Var):
        name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in for statement")
        raise NotImplementedError


def bind_assign_value(self: Parser, _node: doc.expr, var_name: str, value: Any) -> Any:
    if isinstance(value, T.inline):
        return value.value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, _node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        name(var_name, res)
        return res
    elif isinstance(value, (Buffer, IterVar)) or (
        isinstance(value, Var) and not self.var_table.exist(value)
    ):
        name(var_name, value)
        return value
    elif isinstance(value, PrimExpr):
        var = T.var(value.dtype)
        name(var_name, var)
        frame = T.let(var, value)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return var
    return value


@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.frame.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of the following: "
            "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            self.eval_assign(target=node.target, source=iters, bind_value=bind_for_value)
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="While")
def visit_while(self: Parser, node: doc.While) -> None:
    with self.var_table.with_frame():
        cond = self.eval_expr(node.test)
        with T.While(cond):
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AugAssign")
def visit_aug_assign(self: Parser, node: doc.AugAssign) -> None:
    lhs_pos = (
        node.target.lineno,
        node.target.col_offset,
        node.target.end_lineno,
        node.target.end_col_offset,
    )
    rhs_pos = (
        node.value.lineno,
        node.value.col_offset,
        node.value.end_lineno,
        node.value.end_col_offset,
    )
    node.target.ctx = doc.Load(*lhs_pos)
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(*lhs_pos), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(*rhs_pos), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)
    lhs = node.target
    lhs.ctx = doc.Store(*lhs_pos)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_var = self.visit_tvm_annotation(node.annotation)
    if not isinstance(ann_var, Var):
        self.report_error(node.annotation, "Annotation should be Var")
    self.eval_assign(target=lhs, source=ann_var, bind_value=bind_assign_value)
    frame = T.let(ann_var, rhs)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, Frame):
                self.report_error(
                    item.context_expr, "Invalid context expression in the with-statement."
                )
            rhs = stack.enter_context(frame)
            if item.optional_vars is not None:
                self.eval_assign(target=item.optional_vars, source=rhs, bind_value=bind_with_value)
        self.visit_body(node.body)


@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        self.var_table.add("range", T.serial)
        with T.prim_func():
            T.func_name(node.name)
            if node.returns is not None:
                ret_type = self.eval_expr(node.returns)
                if callable(ret_type):
                    ret_type = PrimType(ret_type().dtype)
                T.func_ret(ret_type)
            with self.with_dispatch_token("tir"):
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="tir", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    # TODO: handle different types of arguments:
    # - vararg: arg | None
    # - kwonlyargs: list[arg]
    # - kw_defaults: list[expr | None]
    # - kwarg: arg | None
    # - defaults: list[expr]
    # - posonlyargs: list[arg]
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param = T.arg(arg.arg, self.visit_tvm_annotation(arg.annotation))
        self.var_table.add(arg.arg, param)


@dispatch.register(token="tir", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


@dispatch.register(token="tir", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    res = self.eval_expr(node.value)
    if isinstance(res, Frame):
        res.add_callback(partial(res.__exit__, None, None, None))
        res.__enter__()


@dispatch.register(token="tir", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    with self.var_table.with_frame():
        with T.If(self.eval_expr(node.test)):
            with T.Then():
                self.visit_body(node.body)
            if node.orelse:
                with T.Else():
                    self.visit_body(node.orelse)


@dispatch.register(token="tir", type_name="Assert")
def visit_assert(self: Parser, node: doc.Assert) -> None:
    cond = self.eval_expr(node.test)
    msg = self.eval_expr(node.msg)
    frame = T.Assert(cond, msg)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="Return")
def visit_return(self: Parser, node: doc.Return) -> None:
    self.report_error(node, "Return is not allowed.")
