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
from re import L
import pytest

import tvm
from tvm import relax, tir
from tvm.relax import PyExprVisitor, PyExprMutator
from tvm.ir.base import assert_structural_equal
from tvm.ir import Op
from tvm.relax.ty import DynTensorType
from tvm.relax.expr import Type, Span, Expr
from tvm.relax.expr import Function, ExternFunc
from tvm.relax.expr import Constant, Var, DataflowVar
from tvm.relax.expr import ShapeExpr, RuntimeDepShape
from tvm.relax.expr import GlobalVar, SeqExpr, Tuple
from tvm.relax.expr import Call, If, TupleGetItem
from tvm.relax.expr import Binding, MatchShape, VarBinding
from tvm.relax.expr import BindingBlock, DataflowBlock
from tvm.relax.expr import _update_shape, _update_type

m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
type_anno1 = relax.DynTensorType(1, "float32")
type_anno2 = relax.DynTensorType(2, "float32")
x = relax.Var("x", [n], type_anno1)
y = relax.Var("y", [m, n], type_anno2)
bb = relax.BlockBuilder()


@relax.expr_functor.visitor
class BasicVisitor(PyExprVisitor):
    """Default ExprVisitor"""


class ASTLog:
    def __init__(self, reverse=False) -> None:
        self.log = []
        self.indent = "\t"
        self.level = 0
        self.reverse = reverse

    def push_scope(self):
        self.level += 1

    def pop_scope(self):
        self.level -= 1

    def add(self, s: str):
        self.log.append(self.indent * self.level + s)

    def __str__(self) -> str:
        return "\n".join(reversed(self.log) if self.reverse else self.log)


@relax.expr_functor.visitor
class ASTPrinter(PyExprVisitor):
    """TODO"""

    def __init__(self) -> None:
        self.log = ASTLog()

    def visit_constant_(self, op: Constant) -> None:
        self.log.add("Constant")

    def visit_global_var_(self, op: GlobalVar) -> None:
        self.log.add("GlobalVar")

    def visit_tuple_(self, op: Tuple) -> None:
        self.log.add("Tuple")
        self.log.push_scope()
        for field in op.fields:
            self.visit_expr(field)
        self.log.pop_scope()

    def visit_var_(self, op: Var) -> None:
        self.log.add("Var")

    def visit_dataflow_var_(self, op: DataflowVar) -> None:
        self.log.add("DataflowVar")

    def visit_function_(self, op: Function) -> None:
        self.log.add("Function")
        self.log.push_scope()
        for param in op.params:
            self.visit_var_def(param)

        self.visit_expr(op.body)
        self.log.pop_scope()

    def visit_call_(self, op: Call) -> None:
        self.log.add("Call")
        self.log.push_scope()
        self.visit_expr(op.op)

        for arg in op.args:
            self.visit_expr(arg)
        self.log.pop_scope()

    def visit_if_(self, op: If) -> None:
        self.log.add("If")
        self.log.push_scope()
        self.visit_expr(op.cond)
        self.visit_expr(op.true_branch)
        self.visit_expr(op.false_branch)
        self.log.pop_scope()

    def visit_op_(self, op: Op) -> None:
        self.log.add("Op")

    def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
        self.log.add("TupleGetItem")
        self.log.push_scope()
        self.visit_expr(op.tuple_value)
        self.log.pop_scope()

    def visit_shape_expr_(self, op: ShapeExpr) -> None:
        self.log.add("ShapeExpr")

    def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> None:
        self.log.add("RuntimeDepShape")

    def visit_extern_func_(self, op: ExternFunc) -> None:
        self.log.add("ExternFunc")

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        self.log.add("SeqExpr")
        self.log.push_scope()
        for block in op.blocks:
            self.visit_binding_block(block)
        self.visit_expr(op.body)
        self.log.pop_scope()

    def visit_var_binding_(self, binding: VarBinding) -> None:
        self.log.add("VarBinding")
        self.log.push_scope()
        self.visit_expr(binding.value)
        self.visit_var_def(binding.var)
        self.log.pop_scope()

    def visit_match_shape_(self, binding: MatchShape) -> None:
        self.log.add("MatchShape")
        self.log.push_scope()
        self.visit_expr(binding.value)
        self.visit_expr(ShapeExpr(binding.pattern))
        if binding.var:
            self.visit_var_def(binding.var)
        self.log.pop_scope()

    def visit_binding_block_(self, block: BindingBlock) -> None:
        self.log.add("BindingBlock")
        self.log.push_scope()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.pop_scope()

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        self.log.add("DataflowBlock")
        self.log.push_scope()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.pop_scope()

    def visit_var_def_(self, var: Var) -> None:
        self.log.add("VarDef")

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        self.log.add("DataflowVarDef")


@relax.expr_functor.mutator
class BasicMutator(PyExprMutator):
    """Default ExprMutator"""


@relax.expr_functor.mutator
class ASTPostPrinterMutator(PyExprMutator):
    """TODO"""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def rewrite_constant_post_order(self, op: Constant) -> Expr:
        self.log.add("Constant")
        return op

    def rewrite_global_var_post_order(self, op: GlobalVar) -> Expr:
        self.log.add("GlobalVar")
        return op

    def rewrite_tuple_post_order(self, op: Tuple) -> Expr:
        self.log.add("Tuple")
        return op

    def rewrite_var_post_order(self, op: Var) -> Expr:
        self.log.add("Var")
        return op

    def rewrite_dataflow_var_post_order(self, op: DataflowVar) -> Expr:
        self.log.add("DataflowVar")
        return op

    def rewrite_function_post_order(self, op: Function) -> Expr:
        self.log.add("Function")
        return op

    def rewrite_call_post_order(self, op: Call) -> Expr:
        self.log.add("Call")
        return op

    def rewrite_if_post_order(self, op: If) -> Expr:
        self.log.add("If")
        return op

    def rewrite_op_post_order(self, op: Op) -> Expr:
        self.log.add("Op")
        return op

    def rewrite_tuple_getitem_post_order(self, op: TupleGetItem) -> Expr:
        self.log.add("TupleGetItem")
        return op

    def rewrite_shape_expr_post_order(self, op: ShapeExpr) -> Expr:
        self.log.add("ShapeExpr")
        return op

    def rewrite_runtime_dep_shape_post_order(self, op: RuntimeDepShape) -> Expr:
        self.log.add("RuntimeDepShape")
        return op

    def rewrite_extern_func_post_order(self, op: ExternFunc) -> Expr:
        self.log.add("ExternFunc")
        return op

    def rewrite_seq_expr_post_order(self, op: SeqExpr) -> Expr:
        self.log.add("SeqExpr")
        return op

    def visit_var_binding_(self, binding: VarBinding) -> None:
        new_value = self.visit_expr(binding.value)
        new_var = self.visit_var_def(binding.var)

        def emit(b: VarBinding):
            if self.builder_.current_block_is_dataflow() and not isinstance(b.var, DataflowVar):
                self.builder_.emit_output_var_binding(b)
            else:
                self.builder_.emit_var_binding(b)

        self.log.add("VarBinding")
        if binding.var.same_as(new_var) and binding.value.same_as(new_value):
            emit(binding)
            return

        temp = self.with_shape_and_type(new_var, new_value.shape_, new_value._checked_type_)
        if not temp.same_as(new_var):
            new_var = temp
            self.set_var_remap(binding.var.vid, new_var)

        emit(VarBinding(new_var, new_value))

    def visit_match_shape_(self, binding: MatchShape) -> None:
        new_value = self.visit_expr(binding.value)
        new_pattern = self.visit_expr(ShapeExpr(binding.pattern))

        if binding.var:
            new_shape = None
            if new_value._checked_type_ and isinstance(new_value._checked_type_, DynTensorType):
                new_shape = new_pattern
            new_var = self.visit_var_def(binding.var)
            temp = self.with_shape_and_type(new_var, new_shape, new_value._checked_type_)
            if not temp.same_as(new_var):
                new_var = temp
                self.set_var_remap(binding.var.vid, new_var)

        self.log.add("MatchShape")
        if binding.value.same_as(new_value) and binding.pattern.same_as(new_pattern):
            if not binding.var or (binding.var and binding.var.same_as(new_var)):
                self.builder_.match_shape_binding(binding)
                return

        self.builder_.match_shape_binding(MatchShape(new_value, new_pattern.values, new_var))

    def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
        self.builder_._begin_binding_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("BindingBlock")
        return self.builder_._end_block()

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        self.builder_._begin_dataflow_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("DataflowBlock")
        return self.builder_._end_block()

    def visit_var_def_(self, var: Var) -> None:
        shape_unchanged = True
        new_shape = None
        if var.shape_:
            new_shape = self.visit_expr(var.shape_)
            shape_unchanged &= var.shape_.same_as(new_shape)

        self.log.add("VarDef")
        if shape_unchanged:
            return var
        else:
            new_var = Var(var.vid, None, var._checked_type_, var.span)
            _update_shape(new_var, new_shape)

            self.set_var_remap(var.vid, new_var)
            return new_var

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        shape_unchanged = True
        new_shape = None
        if var.shape_:
            new_shape = self.visit_expr(var.shape_)
            shape_unchanged &= var.shape_.same_as(new_shape)

        self.log.add("DataflowVarDef")
        if shape_unchanged:
            return var
        else:
            new_var = DataflowVar(var.vid, None, var._checked_type_, var.span)
            _update_shape(new_var, new_shape)

            self.set_var_remap(var.vid, new_var)
            return new_var


def basic_check(expr, visitor_str, mutator_str):
    def visit(f, expr):
        if isinstance(expr, relax.Expr):
            return f.visit_expr(expr)
        elif isinstance(expr, relax.BindingBlock):
            return f.visit_binding_block(expr)

    basic_visitor = BasicVisitor()
    visit(basic_visitor, expr)

    log_visitor = ASTPrinter()
    visit(log_visitor, expr)
    assert str(log_visitor.log) == visitor_str

    basic_mutator = BasicMutator()
    if isinstance(expr, relax.Expr):
        expr = bb.normalize(expr)
    assert_structural_equal(visit(basic_mutator, expr), expr)

    post_log_mutator = ASTPostPrinterMutator()
    if isinstance(expr, relax.Expr):
        expr = bb.normalize(expr)
    assert_structural_equal(visit(post_log_mutator, expr), expr)
    assert str(post_log_mutator.log) == mutator_str


def test_constant():
    basic_check(relax.const(1.0), "Constant", "Constant")


def test_var():
    basic_check(x, "Var", "Var")


def test_dataflow_var():
    lv = relax.DataflowVar("lv", [n], type_anno1)
    basic_check(lv, "DataflowVar", "DataflowVar")


def test_tuple():
    t = relax.Tuple([x, y])
    basic_check(t, "\n".join(["Tuple", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Tuple"]))


def test_global_var():
    gv = relax.GlobalVar("gv")
    basic_check(gv, "GlobalVar", "GlobalVar")


def test_seq_expr():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    basic_check(
        seq_expr,
        "\n".join(
            [
                "SeqExpr",
                "\tBindingBlock",
                "\t\tVarBinding",
                "\t\t\tConstant",
                "\t\t\tVarDef",
                "\tVar",
            ]
        ),
        "\n".join(
            ["Constant", "ShapeExpr", "VarDef", "VarBinding", "BindingBlock", "Var", "SeqExpr"]
        ),
    )


def test_shape_expr():
    x = relax.ShapeExpr([m, n])
    basic_check(x, "ShapeExpr", "ShapeExpr")


def test_runtime_dep_shape():
    runtime_dep_shape = relax.RuntimeDepShape()
    basic_check(runtime_dep_shape, "RuntimeDepShape", "RuntimeDepShape")


def test_call():
    call_node = relax.op.add(x, y)
    basic_check(
        call_node,
        "\n".join(["Call", "\tOp", "\tVar", "\tVar"]),
        "\n".join(["Op", "Var", "Var", "Call"]),
    )


def test_if():
    if_node = relax.If(x, x, x)
    basic_check(
        if_node,
        "\n".join(["If", "\tVar", "\tVar", "\tVar"]),
        "\n".join(["Var", "Var", "Var", "If"]),
    )


def test_tuple_getitem():
    tuple_getitem_node = relax.TupleGetItem(relax.Tuple([x, y]), 0)
    basic_check(
        tuple_getitem_node,
        "\n".join(["TupleGetItem", "\tTuple", "\t\tVar", "\t\tVar"]),
        "\n".join(["Var", "Var", "Tuple", "TupleGetItem"]),
    )


def test_binding_block():
    bb._begin_binding_block()
    gv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_shape(y, [m, n])
    b0 = bb._end_block()
    basic_check(
        b0,
        "\n".join(
            [
                "BindingBlock",
                "\tVarBinding",
                "\t\tCall",
                "\t\t\tOp",
                "\t\t\tVar",
                "\t\t\tVar",
                "\t\tVarDef",
                "\tMatchShape",
                "\t\tVar",
                "\t\tShapeExpr",
                "\t\tVarDef",
            ]
        ),
        "\n".join(
            [
                "Op",
                "Var",
                "Var",
                "Call",
                "ShapeExpr",
                "VarDef",
                "VarBinding",
                "Var",
                "ShapeExpr",
                "ShapeExpr",
                "VarDef",
                "MatchShape",
                "BindingBlock",
            ]
        ),
    )


def test_dataflow_block():
    bb._begin_dataflow_block()
    lv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_shape(y, [m, n])
    b0 = bb._end_block()
    basic_check(
        b0,
        "\n".join(
            [
                "DataflowBlock",
                "\tVarBinding",
                "\t\tCall",
                "\t\t\tOp",
                "\t\t\tVar",
                "\t\t\tVar",
                "\t\tDataflowVarDef",
                "\tMatchShape",
                "\t\tVar",
                "\t\tShapeExpr",
                "\t\tDataflowVarDef",
            ]
        ),
        "\n".join(
            [
                "Op",
                "Var",
                "Var",
                "Call",
                "ShapeExpr",
                "DataflowVarDef",
                "VarBinding",
                "Var",
                "ShapeExpr",
                "ShapeExpr",
                "DataflowVarDef",
                "MatchShape",
                "DataflowBlock",
            ]
        ),
    )


def test_function():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    ret_type = relax.DynTensorType(-1, "float32")
    func = relax.Function([x], seq_expr, ret_type)
    basic_check(
        func,
        "\n".join(
            [
                "Function",
                "\tVarDef",
                "\tSeqExpr",
                "\t\tBindingBlock",
                "\t\t\tVarBinding",
                "\t\t\t\tConstant",
                "\t\t\t\tVarDef",
                "\t\tVar",
            ]
        ),
        "\n".join(
            [
                "ShapeExpr",
                "VarDef",
                "Constant",
                "ShapeExpr",
                "VarDef",
                "VarBinding",
                "BindingBlock",
                "Var",
                "SeqExpr",
                "Function",
            ]
        ),
    )


def test_extern_func():
    func = relax.ExternFunc("f")
    basic_check(func, "ExternFunc", "ExternFunc")


if __name__ == "__main__":
    pytest.main([__file__])
