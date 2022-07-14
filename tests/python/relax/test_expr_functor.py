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

import tvm
from tvm import relax, tir
from tvm.relax import ExprFunctor, ExprVisitor, ExprMutatorBase, ExprMutator
from tvm.ir.base import assert_structural_equal

m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
type_anno1 = relax.DynTensorType(1, "float32")
type_anno2 = relax.DynTensorType(2, "float32")
x = relax.Var("x", [n], type_anno1)
y = relax.Var("y", [m, n], type_anno2)
bb = relax.BlockBuilder()


def check_visit(expr):
    def visit(f, expr):
        if isinstance(expr, relax.Expr):
            return f.visit_expr(expr)
        elif isinstance(expr, relax.BindingBlock):
            return f.visit_binding_block(expr)

    if isinstance(expr, relax.Expr):
        with pytest.raises(NotImplementedError):
            ef = ExprFunctor()
            visit(ef, expr)

    ev = ExprVisitor()
    visit(ev, expr)

    em_base = ExprMutatorBase()
    assert_structural_equal(visit(em_base, expr), expr)

    em = ExprMutator()
    if isinstance(expr, relax.Expr):
        expr = bb.normalize(expr)
    assert_structural_equal(visit(em, expr), expr)


def test_constant():
    check_visit(relax.const(1.0))


def test_var():
    check_visit(x)


def test_dataflow_var():
    lv = relax.DataflowVar("lv", [n], type_anno1)
    check_visit(lv)


def test_tuple():
    t = relax.Tuple([x, y])
    check_visit(t)


def test_global_var():
    gv = relax.GlobalVar("gv")
    check_visit(gv)


def test_seq_expr():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    check_visit(seq_expr)


def test_shape_expr():
    x = relax.ShapeExpr([m, n])
    check_visit(x)


def test_runtime_dep_shape():
    runtime_dep_shape = relax.RuntimeDepShape()
    check_visit(runtime_dep_shape)


def test_call():
    call_node = relax.op.add(x, y)
    check_visit(call_node)


def test_if():
    if_node = relax.If(x, x, x)
    check_visit(if_node)


def test_tuple_getitem():
    op = relax.TupleGetItem(relax.Tuple([x, y]), 0)
    check_visit(op)


def test_binding_block():
    bb._begin_binding_block()
    gv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_shape(y, [m, n])
    b0 = bb._end_block()
    check_visit(b0)


def test_dataflow_block():
    bb._begin_dataflow_block()
    lv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_shape(y, [m, n])
    b0 = bb._end_block()
    check_visit(b0)


def test_function():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    ret_type = relax.DynTensorType(-1, "float32")
    func = relax.Function([x], seq_expr, ret_type)
    check_visit(func)


def test_extern_func():
    func = relax.ExternFunc("f")
    check_visit(func)


if __name__ == "__main__":
    pytest.main([__file__])
