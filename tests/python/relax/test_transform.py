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
from tvm import relax
from tvm import tir
from tvm.ir import structural_equal
from tvm.ir.module import IRModule

import tvm.script
from tvm.script import tir as T, relax as R


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
    mod = IRModule.from_expr(expr)

    # before rewrite
    v0 = expr.body.blocks[0].bindings[1].var
    s0 = expr.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.add"
    assert structural_equal(v0.shape, relax.ShapeExpr([m, n]))
    assert structural_equal(s0.shape, relax.ShapeExpr([m, n]))
    assert structural_equal(gv0.shape, relax.ShapeExpr([m, n]))

    # after rewrite
    passes = [relax.transform.FMARewrite()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)
    func = new_mod["main"]
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


def test_to_non_dataflow():
    @tvm.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                lv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                gv0 = relax.call_dps((m, n), "test.op.identity", (lv0,))
                relax.output(gv0)
            return gv0

    mod = TestToNonDataflow

    old_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    x, lv0, gv0 = old_vars

    new_mod = relax.transform.ToNonDataflow()(mod)

    new_vars = []
    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal new_vars
            new_vars.append(e)
    relax.analysis.post_order_visit(new_mod["foo"], fvisit)

    assert x == new_vars[0]
    assert lv0 != new_vars[1]
    assert isinstance(lv0, relax.DataflowVar)
    assert not isinstance(new_vars[1], relax.DataflowVar)

    assert isinstance(gv0, relax.Var)
    assert isinstance(new_vars[2], relax.Var)
    assert gv0 == new_vars[2]


def test_call_dps_rewrite():
    @tvm.script.ir_module
    class TestCallDPSRewrite:
        @R.function
        def foo(x: Tensor[(m, n), "float32"]):
            gv0 = relax.call_dps((m, n), "test.op.identity", (x,))
            return gv0

    mod = TestCallDPSRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.call_dps"

    # after rewrite
    new_mod = relax.transform.CallDPSRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.args[0])
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_vm_memory_lower():
    @tvm.script.ir_module
    class TestVMMemoryLower:
        @R.function
        def foo(x: Tensor[(m, n), "float32"]):
            alloc = relax.builtin.alloc_tensor((m, n))
            _ = relax.call_packed("test.op.identity", (x,), alloc)
            gv0 = alloc
            return gv0
    
    mod = TestVMMemoryLower

    # after vm memory lowering
    new_mod = relax.transform.VMMemoryLower()(mod)
    func = new_mod["foo"]

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(func, tvm.relax.expr.Function)

    block = func.body.blocks[0]
    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.vm.builtin.alloc_storage"
    s2 = block.bindings[1].value
    assert isinstance(s2, tvm.relay.Call)
    s4 = block.bindings[3].value
    assert isinstance(s4, tvm.relay.Call)
    assert isinstance(s4.op, relax.ExternFunc)
    assert s4.op.global_symbol == "test.op.identity"


def test_vm_shape_lowering():
    @tvm.script.ir_module
    class TestVMShapeLower:
        @R.function
        def foo(x: Tensor[_, "float32"]) -> Shape:
            relax.match_shape(x, (n, m))
            return (n * 2, m * 3)

    mod = TestVMShapeLower

    # after vm shape lowering
    new_mod = relax.transform.VMShapeLower()(mod)

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func"], tvm.tir.function.PrimFunc)
    func = new_mod["foo"]
    assert isinstance(func, tvm.relax.expr.Function)

    s1 = func.body.blocks[0].bindings[0].value
    assert isinstance(s1.op, relax.ExternFunc)
    assert s1.op.global_symbol == "vm.builtin.alloc_shape_heap"
    assert s1.args[0].values[0] == 4
    s2 = func.body.blocks[1].bindings[0].value
    assert isinstance(s2.op, relax.ExternFunc)
    assert s2.op.global_symbol == "vm.builtin.shape_of"
    s3 = func.body.blocks[1].bindings[1].value
    assert isinstance(s3, tvm.relay.Call)
    assert s3.op.name == "relax.vm.builtin.store_shape"
    s4 = func.body.blocks[2].bindings[0].value
    assert isinstance(s4.op, relax.GlobalVar)
    assert s4.op.name_hint == "shape_func"
    s5 = func.body.blocks[2].bindings[1].value
    assert isinstance(s5, tvm.relay.Call)
    assert s5.op.name == "relax.vm.builtin.load_shape"


def test_vm_shape_lowering_func_param_with_shape():
    src = """@tvm.script.ir_module
class InputModule:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        m = T.var("int32")
        n = T.var("int32")
        k = T.var("int32")
        A = T.match_buffer(x, (m,n))
        B = T.match_buffer(y, (n,k))
        C = T.match_buffer(z, (m,k))

        for i, j, k in T.grid(m, k, n):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    @R.function
    def foo(x:Tensor[(m, n), "float32"], w:Tensor[(n, k), "float32"]) -> Tensor:
        gv0 = R.call_dps((m, k), tir_matmul, (x, w))
        return gv0
"""
    mod = tvm.script.relax.parser.from_source(src)

    # after vm shape lowering
    new_mod = relax.transform.VMShapeLower()(mod)

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func"], tvm.tir.function.PrimFunc)
    assert isinstance(new_mod["tir_matmul"], tvm.tir.function.PrimFunc)
    func = new_mod["foo"]
    assert isinstance(func, tvm.relax.expr.Function)

    x, w = func.params
    s1 = func.body.blocks[0].bindings[0].value
    assert isinstance(s1.op, relax.ExternFunc)
    assert s1.op.global_symbol == "vm.builtin.alloc_shape_heap"
    assert s1.args[0].values[0] == 3

    s2 = func.body.blocks[0].bindings[1].value
    assert isinstance(s2.op, relax.ExternFunc)
    assert s2.op.global_symbol == "vm.builtin.shape_of"
    assert s2.args[0] == x
    s3 = func.body.blocks[0].bindings[2].value
    assert isinstance(s3, tvm.relay.Call)
    assert s3.op.name == "relax.vm.builtin.store_shape"

    s4 = func.body.blocks[0].bindings[3].value
    assert isinstance(s4.op, relax.ExternFunc)
    assert s4.op.global_symbol == "vm.builtin.shape_of"
    assert s4.args[0] == w
    s5 = func.body.blocks[0].bindings[2].value
    assert isinstance(s5, tvm.relay.Call)
    assert s5.op.name == "relax.vm.builtin.store_shape"


def test_to_anf():
    x = relax.Var("x", type_annotation=relax.DynTensorType())
    gv = relax.op.add(x, x)
    gv1 = relax.op.add(gv, gv)
    gv2 = relax.op.add(gv, gv1)
    body = relax.Tuple([gv, gv2])
    gvar = relax.GlobalVar("f")
    func = relax.Function([x], body, None, gvar)

    mod: tvm.IRModule = tvm.IRModule({gvar: func})
    new_mod = relax.transform.ToANF()(mod)

    @tvm.script.ir_module
    class TestToANFExpected:
        @R.function
        def f(x: Tensor[_, "float32"]):
            gv = relax.add(x, x)
            gv1 = relax.add(gv, gv)
            gv2 = relax.add(gv, gv1)
            return (gv, gv2)

    # TODO(@altanh): fix this once type inference works properly...?
    assert R.parser.astext(new_mod) == R.parser.astext(TestToANFExpected)



if __name__ == "__main__":
    test_fma_rewrite()
    test_to_non_dataflow()
    test_call_dps_rewrite()
    test_vm_memory_lower()
    test_vm_shape_lowering()
    test_vm_shape_lowering_func_param_with_shape()
    test_to_anf()
