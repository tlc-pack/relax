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
from tvm import relax
from tvm import tir
from tvm.ir import structural_equal
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


def test_fma_rewrite():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), y: R.Tensor(("m", "n"), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.add(lv0, y)
                R.output(gv0)
            gv1 = R.multiply(x, y)
            gv2 = R.add(gv1, y)
            return (gv0, gv1, gv2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), y: R.Tensor(("m", "n"), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.ewise_fma(x, y, y)
                R.output(gv0)
            gv1 = R.multiply(x, y)
            gv2 = R.add(gv1, y)
            return (gv0, gv1, gv2)

    After = relax.transform.RewriteFMA()(Before)

    assert_structural_equal(After, Expected)


def test_fma_rewrite_python():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), y: R.Tensor(("m", "n"), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.add(lv0, y)
                R.output(gv0)
            gv1 = R.multiply(x, y)
            gv2 = R.add(gv1, y)
            return (gv0, gv1, gv2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), y: R.Tensor(("m", "n"), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.ewise_fma(x, y, y)
                R.output(gv0)
            gv1 = R.multiply(x, y)
            gv2 = R.add(gv1, y)
            return (gv0, gv1, gv2)

    After = relax.transform.EwiseRewriteFMA()(Before)

    assert_structural_equal(After, Expected)


def test_fma_fuse():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.add(lv0, y)
                R.output(gv0)
            return gv0

    After = relax.transform.FuseFMA()(Before)

    # TODO(@yuchen): add assert_structural_equal after parser allows CallNode as Function's body
    assert len(After.get_global_vars()) == 2
    main = After["main"]
    ewise_fma_fused = After["ewise_fma_fused"]

    # check sub function call type inference
    assert_structural_equal(ewise_fma_fused.body.checked_type, relax.DynTensorType(2, "float32"))
    sub_func_call = main.body.blocks[0].bindings[1].value
    sub_func_call_var = main.body.blocks[0].bindings[1].var
    assert_structural_equal(sub_func_call.checked_type, relax.DynTensorType(2, "float32"))
    assert_structural_equal(sub_func_call_var.checked_type, relax.DynTensorType(2, "float32"))

    # check sub function call shape inference
    assert isinstance(ewise_fma_fused.body.shape, relax.ShapeExpr)
    assert ewise_fma_fused.body.shape.values[0] == 3
    assert ewise_fma_fused.body.shape.values[1] == 4
    assert sub_func_call.shape.values[0] == 3
    assert sub_func_call.shape.values[1] == 4
    assert sub_func_call_var.shape.values[0] == 3
    assert sub_func_call_var.shape.values[1] == 4


def test_fma_fuse_python():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            with R.dataflow():
                lv0 = R.multiply(x, y)
                gv0 = R.add(lv0, y)
                R.output(gv0)
            return gv0

    After = relax.transform.EwiseFuseFMA()(Before)

    # TODO(@yuchen): add assert_structural_equal after parser allows CallNode as Function's body
    assert len(After.get_global_vars()) == 2
    main = After["main"]
    ewise_fma_fused = After["ewise_fma_fused"]

    # check sub function call type inference
    assert_structural_equal(ewise_fma_fused.body.checked_type, relax.DynTensorType(2, "float32"))
    sub_func_call = main.body.blocks[0].bindings[1].value
    sub_func_call_var = main.body.blocks[0].bindings[1].var
    assert_structural_equal(sub_func_call.checked_type, relax.DynTensorType(2, "float32"))
    assert_structural_equal(sub_func_call_var.checked_type, relax.DynTensorType(2, "float32"))

    # check sub function call shape inference
    assert isinstance(ewise_fma_fused.body.shape, relax.ShapeExpr)
    assert ewise_fma_fused.body.shape.values[0] == 3
    assert ewise_fma_fused.body.shape.values[1] == 4
    assert sub_func_call.shape.values[0] == 3
    assert sub_func_call.shape.values[1] == 4
    assert sub_func_call_var.shape.values[0] == 3
    assert sub_func_call_var.shape.values[1] == 4


def test_dataflowpass_fail():
    # raise error on rewriting/removing existing Global Vars inside the dataflow block.
    with pytest.raises(tvm.TVMError):

        @tvm.script.ir_module
        class TestRemoveGlobalScopeVar:
            @R.function
            def main(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
                with R.dataflow():
                    gv_remove = R.add(x, y)
                    gv1 = R.add(x, y)
                    R.output(gv_remove, gv1)
                return (gv_remove, gv1)

        relax.transform.FailTestRewrite()(TestRemoveGlobalScopeVar)

    with pytest.raises(tvm.TVMError):

        @tvm.script.ir_module
        class TestRewriteGlobalScopeVar:
            @R.function
            def main(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
                with R.dataflow():
                    gv_rewrite = R.add(x, y)
                    gv1 = R.add(x, y)
                    R.output(gv_rewrite, gv1)
                return (gv_rewrite, gv1)

        relax.transform.FailTestRewrite()(TestRewriteGlobalScopeVar)

    # raise error on rewriting/removing existing Symbolic Vars inside the dataflow block
    # check all Symbolic Vars defined in R.match_shape
    with pytest.raises(tvm.TVMError):

        @tvm.script.ir_module
        class TestRewriteSymbolicVar:
            @R.function
            def main(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
                with R.dataflow():
                    lv0 = R.match_shape(x, (m, n))
                    gv0 = R.add(lv0, y)
                    R.output(gv0)
                return gv0

        relax.transform.FailTestRewrite()(TestRewriteSymbolicVar)

    with pytest.raises(tvm.TVMError):

        @tvm.script.ir_module
        class TestRemoveSymbolicVar:
            @R.function
            def main(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
                with R.dataflow():
                    lv0 = R.match_shape(x, (m, n, d))
                    gv0 = R.add(lv0, y)
                    R.output(gv0)
                return gv0

        relax.transform.FailTestRewrite()(TestRemoveSymbolicVar)


def test_visit_shape():
    @tvm.script.ir_module
    class TestVisitShape:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            gv0 = R.add(x, x)
            return gv0

    mod = TestVisitShape

    shape_expr = []

    def fvisit(e):
        if isinstance(e, relax.ShapeExpr):
            nonlocal shape_expr
            shape_expr.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)

    # should have visited ShapeExpr 3 times
    # the first time being visited is x.shape
    # the last two times are the call node's shape and gv0's shape
    assert len(shape_expr) == 3
    assert shape_expr[0] == mod["foo"].params[0].shape
    assert shape_expr[1] == shape_expr[2]


def test_to_non_dataflow():
    @tvm.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            with R.dataflow():
                lv0 = R.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
                gv0 = R.call_tir("test.op.identity", (lv0,), (m, n), dtype="float32")
                R.output(gv0)
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


def test_call_tir_rewrite():
    @tvm.script.ir_module
    class TestCallTIRRewrite:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            gv0 = R.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
            return gv0

    mod = TestCallTIRRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.call_tir"

    # after rewrite
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, tvm.relay.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.args[2])
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_vm_memory_lower():
    @tvm.script.ir_module
    class TestVMMemoryLower:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            m, n = T.var("int64"), T.var("int64")
            alloc = R.builtin.alloc_tensor((m, n), runtime_device_index=0, dtype="float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, type_args=(R.Tensor(ndim=2, dtype="float32"))
            )
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
        def foo(x: R.Tensor(dtype="float32")):
            m, n = T.var("int64"), T.var("int64")
            R.match_shape(x, (n, m))
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


def test_vm_static_shape_lowering():
    @tvm.script.ir_module
    class TestVMStaticShapeLower:
        @R.function
        def foo(x: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                y = R.call_tir("test.vm.tile", (x), (2, 6), dtype="float32")
                R.output(y)
            return y

    mod = TestVMStaticShapeLower

    # after vm shape lowering
    new_mod = relax.transform.VMShapeLower()(mod)

    # before and after programs should be structurally equal
    # since the program only has static shapes
    assert_structural_equal(mod, new_mod)


def test_vm_shape_lowering_func_param_with_shape():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")):
            m, k = T.var("int64"), T.var("int64")
            gv0 = R.call_tir(tir_matmul, (x, w), (m, k), dtype="float32")
            return gv0

    mod = InputModule

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


def test_vm_shape_lower_int32_shape():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def foo(x: R.Tensor(("d",), "float32")):
            d = T.var("int64")
            gv0 = R.call_tir("my_extern", (x,), (T.cast(d, "int32"),), dtype="float32")
            return gv0

    before_mod = InputModule
    after_mod = relax.transform.VMShapeLower()(before_mod)

    assert isinstance(after_mod, tvm.IRModule)
    shape_func = after_mod["shape_func"]
    assert isinstance(shape_func, tvm.tir.function.PrimFunc)
    buffer_store_stmt = shape_func.body[0]
    assert isinstance(buffer_store_stmt, tvm.tir.stmt.BufferStore)
    # verify that the value in BufferStore stmt is cast to int64 first
    cast_expr = buffer_store_stmt.value
    assert isinstance(cast_expr, tvm.tir.expr.Cast)
    assert cast_expr.dtype == "int64"


def test_normalize_function():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    type_anno = relax.DynTensorType(ndim=2, dtype="float16")
    x = relax.Var("x", [m, n], type_anno)

    # Note: the parser automatically normalize the IR written in TVMScript,
    # so we manually construct the function here.
    mul_add = relax.Function(
        [x],
        relax.op.multiply(relax.op.add(x, x), relax.op.add(x, x)),
        ret_type=type_anno,
        ret_shape=relax.RuntimeDepShape(),
    )
    mul_add = mul_add.with_attr("global_symbol", "mul_add")
    before_mod = tvm.IRModule.from_expr(mul_add)

    after_mod = relax.transform.Normalize()(before_mod)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def mul_add(x: R.Tensor(("m", "n"), "float16")) -> R.Tensor(dtype="float16", ndim=2):
            gv = R.add(x, x)
            gv1 = R.add(x, x)
            return R.multiply(gv, gv1)

    assert_structural_equal(after_mod, Expected)


def test_normalize_if():
    cond = relax.Var("cond", [], type_annotation=relax.DynTensorType(0, "bool"))
    x = relax.Var("x", [tir.IntImm("int64", 1)], type_annotation=relax.DynTensorType(1, "float32"))
    # TODO(relax-team): add type and shape inference for IfNode
    y = relax.Var("y")

    # Note: the parser automatically normalize the IR written in TVMScript,
    # so we manually construct the function and If here.
    f = relax.Function(
        [cond, x],
        relax.SeqExpr(
            [
                relax.BindingBlock(
                    [
                        relax.VarBinding(
                            y,
                            relax.If(
                                cond,
                                relax.op.multiply(relax.op.add(x, x), relax.op.add(x, x)),
                                relax.op.add(relax.op.multiply(x, x), relax.op.multiply(x, x)),
                            ),
                        )
                    ]
                )
            ],
            y,
        ),
        ret_type=relax.DynTensorType(1, "float32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    f = f.with_attr("global_symbol", "f")
    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def f(
            cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
        ) -> R.Tensor(dtype="float32", ndim=1):
            if cond:
                gv = R.add(x, x)
                gv1 = R.add(x, x)
                y = R.multiply(gv, gv1)
            else:
                gv = R.multiply(x, x)
                gv1 = R.multiply(x, x)
                y = R.add(gv, gv1)
            return y

    assert_structural_equal(after_mod, Expected)


def test_normalize_no_op():
    # the normalize pass should be no-op for IR in ANF
    @tvm.script.ir_module
    class ANFMod1:
        @R.function
        def f(x: R.Tensor(dtype="float32")):
            gv = R.add(x, x)
            gv1 = R.add(gv, gv)
            gv2 = R.add(gv, gv1)
            return (gv, gv2)

    before_mod = ANFMod1
    after_mod = relax.transform.Normalize()(before_mod)
    assert_structural_equal(before_mod, after_mod, map_free_vars=True)

    @tvm.script.ir_module
    class ANFMod2:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            with R.dataflow():
                lv0 = R.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
                gv0 = R.call_tir("test.op.identity", (lv0,), (m, n), dtype="float32")
                R.output(gv0)
            return gv0

    mod = ANFMod2
    mod_post = relax.transform.Normalize()(mod)

    assert_structural_equal(mod, mod_post)


if __name__ == "__main__":
    pytest.main([__file__])
