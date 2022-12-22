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
import pytest
from tvm import relax, tir
from tvm.ir import Op
from tvm.ir.base import assert_structural_equal
from tvm.relax import PyExprMutator, PyExprVisitor
from tvm.relax.expr import (
    BindingBlock,
    Call,
    Constant,
    DataflowBlock,
    DataflowVar,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    If,
    MatchShape,
    RuntimeDepShape,
    SeqExpr,
    ShapeExpr,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
)
from tvm.script import relax as R

m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
x = relax.Var("x", R.Tensor([n], "float32"))
y = relax.Var("y", R.Tensor([m, n], "float32"))
bb = relax.BlockBuilder()


@relax.expr_functor.visitor
class BasicVisitor(PyExprVisitor):
    """Default ExprVisitor"""


class ASTLog:
    """Helper class to log AST"""

    def __init__(self) -> None:
        self.log = []
        self.indent = "\t"
        self.level = 0

    def push_scope(self):
        self.level += 1

    def pop_scope(self):
        self.level -= 1

    def add(self, s: str):
        self.log.append(self.indent * self.level + s)

    def __str__(self) -> str:
        return "\n".join(self.log)


@relax.expr_functor.visitor
class ASTPrinter(PyExprVisitor):
    """Print relax AST in structured format. The shape of Node is ignored."""

    def __init__(self) -> None:
        super().__init__()
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
    """Print relax AST in the post order format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_constant_(self, op: Constant) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Constant")
        return op

    def visit_global_var_(self, op: GlobalVar) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("GlobalVar")
        return op

    def visit_tuple_(self, op: Tuple) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Tuple")
        return op

    def visit_var_(self, op: Var) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Var")
        return op

    def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("DataflowVar")
        return op

    def visit_function_(self, op: Function) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Function")
        return op

    def visit_call_(self, op: Call) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Call")
        return op

    def visit_if_(self, op: If) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("If")
        return op

    def visit_op_(self, op: Op) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Op")
        return op

    def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("TupleGetItem")
        return op

    def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("ShapeExpr")
        return op

    def visit_runtime_dep_shape_(self, op: RuntimeDepShape) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("RuntimeDepShape")
        return op

    def visit_extern_func_(self, op: ExternFunc) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("ExternFunc")
        return op

    def visit_seq_expr_(self, op: SeqExpr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("SeqExpr")
        return op

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Identical with ExprMutator::VisitBinding_(const VarBindingNode* binding) on the C++ side."""
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

        temp = self.with_struct_info(new_var, new_value.struct_info)
        if not temp.same_as(new_var):
            new_var = temp
            self.set_var_remap(binding.var.vid, new_var)

        emit(VarBinding(new_var, new_value))

    def visit_match_shape_(self, binding: MatchShape) -> None:
        """Identical with ExprMutator::VisitBinding_(const MatchShapeNode* binding) on the C++ side."""
        new_value = self.visit_expr(binding.value)
        new_pattern = self.visit_expr(ShapeExpr(binding.pattern))

        if binding.var:
            new_sinfo = None
            if isinstance(new_value.struct_info, TensorStructInfo):
                new_sinfo = relax.TensorStructInfo(new_pattern, dtype=new_value.struct_info)
            else:
                new_sinfo = new_value.struct_info

            new_var = self.visit_var_def(binding.var)
            temp = self.with_struct_info(new_var, new_sinfo)
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
        """Identical with ExprMutator::VisitBindingBlock_(const BindingBlockNode* block) on the C++ side."""
        self.builder_._begin_binding_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("BindingBlock")
        return self.builder_._end_block()

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        """Identical with ExprMutator::VisitBindingBlock_(const DataflowBlockNode* block) on the C++ side."""
        self.builder_._begin_dataflow_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("DataflowBlock")
        return self.builder_._end_block()

    def visit_var_def_(self, var: Var) -> None:
        """Identical with ExprMutator::VisitVarDef_(const VarNode* var) on the C++ side."""
        shape_unchanged = True
        new_shape = None
        if var.shape_:
            new_shape = self.visit_expr(var.shape_)
            shape_unchanged &= var.shape_.same_as(new_shape)

        self.log.add("VarDef")
        if shape_unchanged:
            return var
        else:
            new_var = Var(var.vid, new_shape, var._checked_type_, var.span)

            self.set_var_remap(var.vid, new_var)
            return new_var

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        """Identical with ExprMutator::VisitVarDef_(const DataflowVarNode* var) on the C++ side."""
        shape_unchanged = True
        new_shape = None
        if var.shape_:
            new_shape = self.visit_expr(var.shape_)
            shape_unchanged &= var.shape_.same_as(new_shape)

        self.log.add("DataflowVarDef")
        if shape_unchanged:
            return var
        else:
            new_var = DataflowVar(var.vid, new_shape, var._checked_type_, var.span)

            self.set_var_remap(var.vid, new_var)
            return new_var


def basic_check(expr, visitor_str, mutator_str):
    def visit(f, expr):
        if isinstance(expr, relax.Expr):
            return f.visit_expr(expr)
        elif isinstance(expr, relax.BindingBlock):
            return f.visit_binding_block(expr)

    # check no overloading case
    basic_visitor = BasicVisitor()
    visit(basic_visitor, expr)

    # check the output log
    log_visitor = ASTPrinter()
    visit(log_visitor, expr)
    assert str(log_visitor.log) == visitor_str

    # check no overloading case
    basic_mutator = BasicMutator()
    # skip normalize GlobalVar since it requires context IRModule to get the checked_type_
    if isinstance(expr, relax.Expr) and not isinstance(expr, relax.GlobalVar):
        expr = bb.normalize(expr)
        assert_structural_equal(visit(basic_mutator, expr), expr)

    # check the output log and return value
    post_log_mutator = ASTPostPrinterMutator()
    if isinstance(expr, relax.Expr) and not isinstance(expr, relax.GlobalVar):
        expr = bb.normalize(expr)
        assert_structural_equal(visit(post_log_mutator, expr), expr)
        assert str(post_log_mutator.log) == mutator_str


def test_constant():
    basic_check(relax.const(1.0), "Constant", "Constant")


def test_var():
    basic_check(x, "Var", "Var")


@pytest.mark.skip("Revisit PyMutator tests after struct info")
def test_dataflow_var():
    lv = relax.DataflowVar("lv", R.Tensor([n], "float32"))
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
        "\n".join(["Var", "Var", "SeqExpr", "Var", "SeqExpr", "If"]),
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
    func = relax.Function([x], seq_expr, R.Tensor([n], "float32"))
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


def test_inherit():
    # The internal class is not instantiated.
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

    call_node = relax.op.add(x, y)
    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "\tOp", "\tVar", "\tVar"])


def test_inherit_with_cls():
    # The decorator converts `InternalVisitor` to a wrapper class.
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    # `InternalVisitor._cls` refers to the original `InternalVisitor` users defined.
    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

    call_node = relax.op.add(x, y)
    iv = InternalVisitor()
    iv.visit_expr(call_node)
    assert str(iv.log) == "\n".join(["InternalCall", "\tOp", "\tVar", "\tVar"])

    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "\tOp", "\tVar", "\tVar"])


def test_wrong_inherit():
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def visit_call_(self, op: Call) -> None:
            pass

    with pytest.raises(
        TypeError,
        match="Inheritance from a decorated object `LeafVisitor` is not allowed. Please inherit from `LeafVisitor._cls`.",
    ):

        @relax.expr_functor.visitor
        class LeafVisitor(InternalVisitor):
            def visit_call_(self, op: Call) -> None:
                pass


def test_call_visitor_super():
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            super().visit_call_(op)  # call PyExprVisitor.visit_call_

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            super().visit_call_(op)  # call InternalVisit.visit_call_

    call_node = relax.op.add(x, y)
    iv = InternalVisitor()
    iv.visit_expr(call_node)
    assert str(iv.log) == "\n".join(["InternalCall", "Op", "Var", "Var"])

    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "InternalCall", "Op", "Var", "Var"])


def test_call_mutator_super():
    @relax.expr_functor.mutator
    class InternalMutator(PyExprMutator):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            return super().visit_call_(op)  # call PyExprMutator.visit_call_

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")
            return super().visit_var_(op)  # call PyExprMutator.visit_var_

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")
            return super().visit_op_(op)  # call PyExprMutator.visit_op_

    @relax.expr_functor.mutator
    class LeafMutator(InternalMutator._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            return super().visit_call_(op)  # call InternalMutator.visit_call_

    call_node = relax.op.add(x, y)
    im = InternalMutator()
    im.visit_expr(call_node)
    assert str(im.log) == "\n".join(["InternalCall", "Op", "Var", "Var"])

    lm = LeafMutator()
    lm.visit_expr(call_node)
    assert str(lm.log) == "\n".join(["LeafCall", "InternalCall", "Op", "Var", "Var"])


if __name__ == "__main__":
    pytest.main([__file__])
