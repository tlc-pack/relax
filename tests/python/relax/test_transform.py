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

from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import tir
from tvm import relax
from tvm.ir import structural_equal

import tvm.script
from tvm.script import relax as R

import numpy as np


def test_fma_rewrite():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = relax.DynTensorType(rank=2, dtype="float16")
    dtype1 = relax.DynTensorType(rank=2, dtype="float16")
    x = relax.Var("x", [m, n], dtype0)
    y = relax.Var("y", [m, n], dtype1)
    ib = relax.BlockBuilder()
    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(relax.op.multiply(x, y))
            gv0 = ib.emit_output(relax.op.add(lv0, y))
        ib.emit_func_output(gv0)
    expr = ib.get()

    # before rewrite
    v0 = expr.body.blocks[0].bindings[1].var
    s0 = expr.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.add"
    assert structural_equal(v0.shape, relax.ShapeExpr([m, n]))
    assert structural_equal(s0.shape, relax.ShapeExpr([m, n]))
    assert structural_equal(gv0.shape, relax.ShapeExpr([m, n]))

    # after rewrite
    func = relax.transform.fma_rewrite(expr)
    v1 = func.body.blocks[0].bindings[1].var
    s1 = func.body.blocks[0].bindings[1].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.ewise_fma"
    assert structural_equal(v1.shape, relax.ShapeExpr([m, n]))
    assert structural_equal(s1.shape, relax.ShapeExpr([m, n]))

    # The var binded to the fma call is reused because the shape
    # and type of var are unchanged after rewriting
    assert gv0 == v0
    assert type(func.body.blocks[0].bindings[1].var) == relax.Var


def test_call_dps_rewrite():
    @rx.script
    class TestCallDpsRewrite:
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                lv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                gv1 = relax.call_dps((m, n), "test.op.identity", (lv0,))
                relax.output(gv1)
            return gv1

    mod = TestCallDpsRewrite()

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    _, x, _, gv0, _, gv1 = old_vars

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.call_dps"

    # after rewrite
    new_mod = rx.transform.call_dps_rewrite(mod)

    new_vars = []
    def fvisit(e):
        if isinstance(e, rx.Var):
            nonlocal new_vars
            new_vars.append(e)
    rx.analysis.post_order_visit(new_mod["foo"], fvisit)

    # the dataflow block has changed to binding block due to the rewriting
    block = new_mod["foo"].body.blocks[0]
    assert not isinstance(block, rx.DataflowBlock)

    # all dataflow vars have changed to vars after the rewriting
    assert x == new_vars[0]
    assert isinstance(gv0, rx.DataflowVar)
    assert not isinstance(gv1, rx.DataflowVar)
    for var in new_vars:
        assert not isinstance(var, rx.DataflowVar)

    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.args[0])
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_memory_lower():
    @tvm.script.ir_module
    class TestMemoryLower:
        @R.function
        def foo(x: Tensor[(m, n), "float32"]):
            alloc = relax.builtin.alloc_tensor((m, n))
            _ = relax.call_packed("test.op.identity", (x,), alloc)
            gv0 = alloc
            return gv0

    mod = TestMemoryLower

    # after memory lowering
    new_mod = relax.transform.memory_lower(mod)

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = R.parser.astext(new_mod)
    assert "vm.builtin.alloc_storage" in code
    assert "vm.builtin.alloc_tensor" in code


def test_shape_lowering():
    @tvm.script.ir_module
    class TestShapeLower:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Shape:
            sh = relax.call_packed("vm.builtin.shape_of", x)
            relax.match_shape(sh, (n, m))
            return (n * 2, m * 3)

    mod = TestShapeLower
    new_mod = relax.transform.shape_lower(mod)
    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func"], tvm.tir.function.PrimFunc)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = R.parser.astext(new_mod)
    assert "alloc_shape_heap" in code
    assert "decode_shape" in code
    assert "make_shape" in code


if __name__ == "__main__":
    test_fma_rewrite()
    test_call_dps_rewrite()
    test_memory_lower()
    test_shape_lowering()
