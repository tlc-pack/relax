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
import tvm
from tvm import tir
from tvm import relax as rx
from tvm.ir import structural_equal

def test_dispatch_var():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    v0 = rx.Var("v0", [m, n], dtype0)
    v1 = rx.DataflowVar("v1", [n], dtype1)
    t = None
    def fvisit(e):
        nonlocal t
        t = type(e)
    rx.analysis.post_order_visit(v0, fvisit)
    assert t == type(v0)
    rx.analysis.post_order_visit(v1, fvisit)
    assert t == type(v1)


def test_post_order_visit():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.IRBuilder()
    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_output(gv0)
    expr = ib.get()

    names = []
    def fvisit(e):
        nonlocal names
        if isinstance(e, tvm.ir.op.Op):
            names.append(e.name)
    rx.analysis.post_order_visit(expr.body, fvisit)
    assert names == ["relax.add", "relax.multiply"]


def test_fma_rewrite():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=2, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [m, n], dtype1)
    ib = rx.IRBuilder()
    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.multiply(x, y))
            lv1 = ib.emit(rx.op.add(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_output(gv0)
    expr = ib.get()

    # before rewrite
    v0 = expr.body.blocks[0].bindings[1].var
    s0 = expr.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.add"
    assert structural_equal(v0.shape, rx.ShapeExpr([m, n]))
    assert structural_equal(s0.shape, rx.ShapeExpr([m, n]))
    assert structural_equal(gv0.shape, rx.ShapeExpr([m, n]))

    # after rewrite
    func = rx.analysis.fma_rewrite(expr)

    v1 = func.body.blocks[0].bindings[1].var
    s1 = func.body.blocks[0].bindings[1].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.ewise_fma"
    assert structural_equal(v1.shape, rx.ShapeExpr([m, n]))
    assert structural_equal(s1.shape, rx.ShapeExpr([m, n]))
    
    # The var binded to the fma call is reused because the shape 
    # and type of var are unchanged after rewriting
    assert lv1 == v0

    assert type(func.body.blocks[0].bindings[2].var) == rx.Var
    assert type(func.body.blocks[0].bindings[2].value) == rx.DataflowVar

def test_lazy_irbuilder():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=2, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [m, n], dtype1)
    ib = rx.IRBuilder()

    # This program should not be rewritten by the fma_rewriter
    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.multiply(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_output(gv0)
    expr = ib.get()

    # before rewrite
    block0 = expr.body.blocks[0]
    v0 = expr.body.blocks[0].bindings[1].var
    s0 = expr.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.multiply"

    # after rewrite (the bindings and the dataflow block are reused)
    func = rx.analysis.fma_rewrite(expr)

    block1 = func.body.blocks[0]
    v1 = func.body.blocks[0].bindings[1].var
    s1 = func.body.blocks[0].bindings[1].value
   
    # the dataflow block and vars are reused
    assert block0 == block1
    assert v1 == v0
    assert s1 == s0

def test_explicit_memory_rewrite():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    shape_anno = [m, n]
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", shape_anno, type_anno)
    ib = rx.IRBuilder()
    with ib.function(x):
        with ib.dataflow() as df:
            lv0 = rx.call_dps([m, n], rx.extern("test.op.identity"), [x])
            gv0 = ib.emit_output(lv0)
        ib.emit_output(gv0)
    expr = ib.get()

    # before rewrite
    v0 = expr.body.blocks[0].bindings[0].var
    s0 = expr.body.blocks[0].bindings[0].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.call_dps"

    # after rewrite
    func = rx.analysis.explicit_memory_rewrite(expr)

    # the dataflow block has changed to binding block due to the rewriting
    block = func.body.blocks[0]
    assert isinstance(block, rx.BindingBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], rx.ShapeExpr)
    assert structural_equal(s1.args[0], rx.ShapeExpr(shape_anno))
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


if __name__ == "__main__":
    test_dispatch_var()
    test_post_order_visit()
    test_fma_rewrite()
    test_lazy_irbuilder()
    test_explicit_memory_rewrite()
