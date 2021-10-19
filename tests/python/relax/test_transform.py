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
from tvm import relax as rx
from tvm.ir import structural_equal
import numpy as np


def test_fma_rewrite():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=2, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [m, n], dtype1)
    ib = rx.BlockBuilder()
    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.multiply(x, y))
            gv0 = ib.emit_output(rx.op.add(lv0, y))
        ib.emit_func_output(gv0)
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
    func = rx.transform.fma_rewrite(expr)
    v1 = func.body.blocks[0].bindings[1].var
    s1 = func.body.blocks[0].bindings[1].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.ewise_fma"
    assert structural_equal(v1.shape, rx.ShapeExpr([m, n]))
    assert structural_equal(s1.shape, rx.ShapeExpr([m, n]))

    # The var binded to the fma call is reused because the shape
    # and type of var are unchanged after rewriting
    assert gv0 == v0
    assert type(func.body.blocks[0].bindings[1].var) == rx.Var


def test_call_dps_rewrite():
    @rx.script
    class Mod:
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                gv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                relax.output(gv0)
            return gv0

    mod = Mod()
    code = rx.parser.astext(mod)

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.call_dps"

    # after rewrite
    new_mod = rx.transform.call_dps_rewrite(mod)
    func = new_mod["foo"]
    code = rx.parser.astext(new_mod)

    # the dataflow block has changed to binding block due to the rewriting
    block = func.body.blocks[0]
    assert not isinstance(block, rx.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], rx.ShapeExpr)
    # assert structural_equal(s1.args[0], rx.ShapeExpr(shape_anno))
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"
    return new_mod


def test_memory_lower():
    @rx.script
    class Mod:
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                gv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                relax.output(gv0)
            return gv0

    mod = test_call_dps_rewrite()
    code = rx.parser.astext(mod)

    # after memory lowering
    new_mod = rx.transform.memory_lower(mod)

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = rx.parser.astext(new_mod)
    assert "vm.builtin.alloc_storage" in code
    assert "vm.builtin.alloc_tensor" in code


def test_shape_lowering():
    @rx.script
    class Mod1:
        def foo(x: Tensor[_, "float32"]) -> Shape:
            sh = relax.call_packed("vm.builtin.shape_of", x)
            relax.match_shape(sh, (n, m))
            return (n * 2, m * 3)

    mod = Mod1()
    new_mod = rx.transform.shape_lower(mod)
    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func"], tvm.tir.function.PrimFunc)
    assert isinstance(new_mod["foo"], tvm.relax.expr.Function)
    code = rx.parser.astext(new_mod)
    assert "alloc_shape_heap" in code
    assert "decode_shape" in code
    assert "make_shape" in code


if __name__ == "__main__":
    test_fma_rewrite()
    test_call_dps_rewrite()
    test_memory_lower()
    test_shape_lowering()
