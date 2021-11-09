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
from tvm.ir import structural_equal

import tvm.script
from tvm.script import relax as R


def test_fma_rewrite():
    @tvm.script.ir_module
    class TestFmaRewrite:
        @R.function
        def foo(x: Tensor[(m, n), "float32"], y: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                gv0 = relax.add(lv0, y)
                relax.output(gv0)
            return gv0

    mod = TestFmaRewrite

    # before rewrite
    func = mod["foo"]
    s0 = func.body.blocks[0].bindings[1].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.add"

    # after rewrite
    passes = [relax.transform.FMARewrite()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)
    func = new_mod["foo"]
    s1 = func.body.blocks[0].bindings[1].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.ewise_fma"


def test_to_non_dataflow():
    @tvm.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: Tensor[(m, n), "float32"]):
            with relax.dataflow():
                gv0 = relax.call_dps((m, n), "test.op.identity", (x,))
                gv1 = relax.call_dps((m, n), "test.op.identity", (gv0,))
                relax.output(gv1)
            return gv1

    mod = TestToNonDataflow

    old_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    _, x, _, gv0, _, gv1 = old_vars

    passes = [relax.transform.ToNonDataflow()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)

    new_vars = []
    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal new_vars
            new_vars.append(e)
    relax.analysis.post_order_visit(new_mod["foo"], fvisit)

    assert x == new_vars[1]
    assert gv0 != new_vars[3]
    assert isinstance(gv0, relax.DataflowVar)
    assert not isinstance(new_vars[3], relax.DataflowVar)

    assert isinstance(gv1, relax.Var)
    assert isinstance(new_vars[5], relax.Var)
    assert gv1 == new_vars[5]


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
    passes = [relax.transform.CallDPSRewrite()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)
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
    passes = [relax.transform.VMMemoryLower()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)
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
            sh = relax.call_packed("vm.builtin.shape_of", x)
            relax.match_shape(sh, (n, m))
            return (n * 2, m * 3)

    mod = TestVMShapeLower

    # after vm shape lowering
    passes = [relax.transform.VMShapeLower()]
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(new_mod["shape_func"], tvm.tir.function.PrimFunc)
    func = new_mod["foo"]
    assert isinstance(func, tvm.relax.expr.Function)

    s1 = func.body.blocks[0].bindings[0].value
    assert isinstance(s1.op, relax.ExternFunc)
    assert s1.op.global_symbol == "vm.builtin.alloc_shape_heap"
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

def test_to_anf():
    x = relax.Var("x", type_annotation=relax.DynTensorType())
    gv = relax.op.add(x, x)
    gv1 = relax.op.add(gv, gv)
    gv2 = relax.op.add(gv, gv1)
    body = relax.Tuple([gv, gv2])
    gvar = relax.GlobalVar("f")
    func = relax.Function([x], body, None, gvar)

    mod: tvm.IRModule = tvm.IRModule({gvar: func})
    mod = relax.transform.to_anf(mod)

    @tvm.script.ir_module
    class TestToANFExpected:
        @R.function
        def f(x: Tensor[_, "float32"]):
            gv = relax.add(x, x)
            gv1 = relax.add(gv, gv)
            gv2 = relax.add(gv, gv1)
            return (gv, gv2)

    # TODO(@altanh): fix this once type inference works properly...?
    assert R.parser.astext(mod) == R.parser.astext(TestToANFExpected)



if __name__ == "__main__":
    test_fma_rewrite()
    test_to_non_dataflow()
    test_call_dps_rewrite()
    test_vm_memory_lower()
    test_vm_shape_lowering()
