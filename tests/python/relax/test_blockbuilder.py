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
from tvm import tir, te
from tvm import relay
from tvm import relax as rx
from tvm.tir.function import PrimFunc

from tvm.ir.base import assert_structural_equal
from tvm.relax import ExternFunc
from tvm import topi
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T


@tvm.register_func("test.blockbuilder.nop")
def nop():
    pass


def test_block_builder():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    bb._begin_binding_block()
    gv0 = bb.emit(rx.op.add(x, y))
    bb._begin_dataflow_block()
    lv0 = bb.emit(rx.op.multiply(gv0, y))
    gv1 = bb.emit_output(rx.op.multiply(lv0, lv0))
    b0 = bb._end_block()
    bb._begin_dataflow_block()
    lv1 = bb.emit(rx.op.multiply(gv0, y))
    gv2 = bb.emit_output(rx.op.multiply(lv1, lv1))
    b1 = bb._end_block()
    gv3 = bb.emit(rx.op.add(x, y))
    b2 = bb._end_block()

    assert isinstance(b0, rx.DataflowBlock)
    assert isinstance(b1, rx.DataflowBlock)
    assert not isinstance(b2, rx.DataflowBlock)


def test_function_single_block():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            lv1 = bb.emit(rx.op.multiply(lv0, y))
            assert lv1.name_hint == "lv1"
            gv0 = bb.emit_output(lv1)
        assert gv0.name_hint == "gv"
        bb.emit_func_output(gv0)

    func = bb.get()["func"]
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv0
    assert gv0.shape[0] == m
    assert gv0.shape[1] == n
    assert gv0.checked_type.ndim == 2
    assert gv0.checked_type.dtype == "float16"
    assert len(func.body.blocks) == 1
    assert len(func.body.blocks[0].bindings) == 3


def test_function_multi_blocks():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            gv0 = bb.emit_output(lv0)
        assert gv0.name_hint == "gv"
        gv1 = bb.emit(rx.op.add(gv0, gv0))
        assert gv1.name_hint == "gv1"
        with bb.dataflow():
            lv1 = bb.emit(rx.op.add(gv1, gv1))
            assert lv1.name_hint == "lv1"
            gv2 = bb.emit_output(gv1)
        bb.emit_func_output(gv2)

    func = bb.get()["func"]
    assert gv2.shape[0] == m
    assert gv2.shape[1] == n
    assert gv2.checked_type.ndim == 2
    assert gv2.checked_type.dtype == "float16"
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv2
    assert len(func.body.blocks) == 3
    assert len(func.body.blocks[0].bindings) == 2
    assert len(func.body.blocks[1].bindings) == 1
    assert len(func.body.blocks[2].bindings) == 2


def test_multi_functions():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func1", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)

    with bb.function("func2", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(y, x))
            # TODO(@yuchen): enable block builder to reset local var unique name map
            assert lv0.name_hint == "lv1"
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)

    mod = bb.get()
    func1 = mod["func1"]
    assert func1.params[0] == x
    assert func1.params[1] == y
    assert len(func1.body.blocks) == 1
    func2 = mod["func2"]
    assert func2.params[0] == x
    assert func2.params[1] == y
    assert len(func2.body.blocks) == 1


def test_block_builder_input_mod():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
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
        def before_main(
            x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")
        ) -> R.Tensor:
            m, n, k = T.var("int64"), T.var("int64"), T.var("int64")
            gv0 = R.call_tir("tir_matmul", (x, w), (m, k), dtype="float32")
            return gv0

    @R.function
    def after_main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        gv0 = R.call_tir("tir_matmul", (x, w), (32, 32), dtype="float32")
        return gv0

    input_mod = InputModule
    bb = rx.BlockBuilder(input_mod)
    var_main = input_mod.get_global_var("before_main")
    bb.update_func(var_main, after_main)

    context_mod = bb.get()
    assert len(context_mod.get_global_vars()) == 2
    var_before_main = context_mod.get_global_var("before_main")
    assert var_main == var_before_main
    assert_structural_equal(context_mod[var_before_main], after_main)


def test_binary_shape_type_deduction():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    k = tir.Var("k", "int64")
    x = rx.Var("x", R.Tensor([m, 1], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    z = rx.Var("z", R.Tensor([5], "float16"))
    w = rx.Var("w", R.Tensor([k], "float16"))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y, z, w]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.shape[0] == m
            assert lv0.shape[1] == n
            assert isinstance(lv0.checked_type, rx.DynTensorType)
            assert lv0.checked_type.ndim == 2
            assert lv0.checked_type.dtype == "float16"

            lv1 = bb.emit(rx.op.multiply(x, z))
            assert lv1.shape[0] == m
            assert lv1.shape[1] == 5
            assert isinstance(lv1.checked_type, rx.DynTensorType)
            assert lv1.checked_type.ndim == 2
            assert lv1.checked_type.dtype == "float16"

            lv2 = bb.emit(rx.op.multiply(z, w))
            assert isinstance(lv2.checked_type, rx.DynTensorType)
            assert lv2.checked_type.ndim == 1
            assert lv2.checked_type.dtype == "float16"

            lv3 = bb.emit(rx.op.multiply(y, w))
            assert isinstance(lv3.struct_info, rx.TensorStructInfo)
            assert lv3.checked_type.ndim == 1
            assert lv3.checked_type.dtype == "float16"
            gv0 = bb.emit_output(lv3)
        bb.emit_func_output(gv0)

        assert isinstance(gv0.checked_type, rx.DynTensorType)
        assert gv0.checked_type.ndim == 1
        assert gv0.checked_type.dtype == "float16"


def test_emit_match_shape():
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    x = rx.Var("tensor_value", R.Tensor("float32", ndim=-1))
    y = rx.Var("shape_value", R.Shape([16, 8]))
    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            # lv0: Tensor((m, n), "float32") =
            #   match_shape(x: Tensor(_, "float32"], [m, n))
            lv0 = bb.match_shape(x, [m, n])
            assert isinstance(lv0, rx.DataflowVar)
            assert lv0.shape[0] == m
            assert lv0.shape[1] == n
            assert lv0.checked_type.ndim == 2
            assert lv0.checked_type.dtype == "float32"

            # lv1: Shape = match_shape(shape, [m, n])
            lv1 = bb.match_shape(y, [m, n])
            assert lv1.checked_type == rx.ShapeType(2)
            gv0 = bb.emit_output(lv1)

        bb.emit_func_output(gv0)
    func = bb.get()["func"]
    block = func.body.blocks[0]
    b0, b1 = block.bindings[:2]
    assert isinstance(b0, rx.MatchShape)
    assert isinstance(b1, rx.MatchShape)

    assert b0.value == x
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.var == lv0

    assert b1.value == y
    assert b1.pattern[0] == m
    assert b1.pattern[1] == n
    assert b1.var == lv1


def test_emit_match_shape_binding_in_dataflow_block():
    bb = rx.BlockBuilder()

    x = rx.Var("x", R.Tensor("float32", ndim=-1))
    m = tir.Var("m", dtype="int64")
    gv = rx.Var("gv", R.Tensor("float32", ndim=-1))
    match_shape = rx.MatchShape(x, (m,), gv)

    with bb.function("main", [x]):
        with bb.dataflow():
            bb.match_shape_binding(match_shape)
            bb.emit_output(gv)
        bb.emit_func_output(x)

    func = bb.get()["main"]
    block = func.body.blocks[0]
    b0 = block.bindings[0]
    assert isinstance(b0, rx.MatchShape)

    assert b0.value == x
    assert b0.pattern[0] == m
    assert b0.var == gv


def test_normalize():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    type_anno0 = rx.DynTensorType(ndim=2, dtype="float16")
    type_anno1 = rx.DynTensorType(ndim=1, dtype="float16")

    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    # Call node
    add_call = rx.op.multiply(x, y)
    assert isinstance(add_call.shape, rx.Call)

    bb.normalize(add_call)
    assert isinstance(add_call.shape, rx.ShapeExpr)
    assert add_call.shape[0] == m
    assert add_call.shape[1] == n

    # Tuple node
    tuple_1 = rx.Tuple([x, y])
    bb.normalize(tuple_1)
    assert_structural_equal(tuple_1.checked_type, rx.TupleType([type_anno0, type_anno1]))
    assert_structural_equal(tuple_1.shape, rx.Tuple([x.shape, y.shape]))
    assert isinstance(tuple_1.struct_info, rx.TupleStructInfo)
    assert isinstance(tuple_1.struct_info.fields[0], rx.TensorStructInfo)
    assert isinstance(tuple_1.struct_info.fields[1], rx.TensorStructInfo)

    # Note sure if it's needed
    assert_structural_equal(
        tuple_1.shape.struct_info,
        rx.TupleStructInfo([rx.ShapeStructInfo([m, n]), rx.ShapeStructInfo([n])]),
    )

    # Nested Tuple
    tuple_2 = rx.Tuple([x, rx.Tuple([x, y])])
    bb.normalize(tuple_2)
    assert_structural_equal(
        tuple_2.checked_type, rx.TupleType([type_anno0, rx.TupleType([type_anno0, type_anno1])])
    )
    assert_structural_equal(tuple_2.shape, rx.Tuple([x.shape, rx.Tuple([x.shape, y.shape])]))
    assert isinstance(tuple_2.struct_info, rx.TupleStructInfo)
    assert isinstance(tuple_2.struct_info.fields[0], rx.TensorStructInfo)
    assert isinstance(tuple_2.struct_info.fields[1], rx.TupleStructInfo)
    assert isinstance(tuple_2.struct_info.fields[1].fields[0], rx.TensorStructInfo)
    assert isinstance(tuple_2.struct_info.fields[1].fields[1], rx.TensorStructInfo)
    # assert_structural_equal(
    #     tuple_2.shape.checked_type,
    #     rx.TupleType([rx.ShapeType(), rx.TupleType([rx.ShapeType(), rx.ShapeType()])]),
    # )


def test_call_te():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    y = rx.Var("y", R.Tensor([n, m], "float32"))
    z = rx.Var("z", R.Tensor([n, m], "float32"))

    def te_func(args, args_dict, msg):
        A, B = args
        C = args_dict["C"]
        D = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
        E = te.compute((128, 128), lambda i, j: D[i, j] - C[i, j])
        return E

    with bb.function("rx_func", [x, y, z]):
        with bb.dataflow():
            out = bb.emit_output(bb.call_te(te_func, [x, y], {"C": z}, msg="hello"))
        bb.emit_func_output(out)

    mod = bb.get()
    rx_func = mod["rx_func"]

    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert rx_func.params[2] == z
    assert rx_func.body.body == out
    assert len(rx_func.body.blocks) == 1
    assert len(rx_func.body.blocks[0].bindings) == 1


def test_emit_te():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    y = rx.Var("y", R.Tensor([n, m], "float32"))
    z = rx.Var("z", R.Tensor([n, m], "float32"))

    def te_func(args, args_dict, msg):
        A, B = args
        C = args_dict["C"]
        D = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
        E = te.compute((128, 128), lambda i, j: D[i, j] - C[i, j])
        return E

    with bb.function("rx_func", [x, y, z]):
        out = bb.emit_te(te_func, [x, y], {"C": z}, msg="hello")
        bb.emit_func_output(out)

    mod = bb.get()
    rx_func = mod["rx_func"]

    def get_tir_func():
        A = te.placeholder((n, m), dtype="float32", name="A")
        B = te.placeholder((n, m), dtype="float32", name="B")
        C = te.placeholder((n, m), dtype="float32", name="C")
        out = te_func((A, B), {"C": C}, "")
        return tvm.te.create_prim_func([A, B, C, out], index_dtype_override="int64")

    # check TIR structure matches expected
    assert_structural_equal(mod["te_func"].body, get_tir_func().body)

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert rx_func.params[2] == z
    assert rx_func.body.body == out
    assert len(rx_func.body.blocks) == 1
    assert len(rx_func.body.blocks[0].bindings) == 1

    call_node = rx_func.body.blocks[0].bindings[0].value
    assert isinstance(call_node, rx.Call)
    assert call_node.op == relay.op.get("relax.call_tir")
    assert len(call_node.args) == 3
    assert call_node.args[0].name_hint == "te_func"
    assert call_node.args[1][0] == x
    assert call_node.args[1][1] == y
    assert call_node.args[1][2] == z


def test_emit_te_multiple():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    y = rx.Var("y", R.Tensor([n, m], "float32"))
    z = rx.Var("z", R.Tensor([128, m], "float32"))

    def te_func(A):
        B = te.compute((128, 128), lambda i, j: A[i, j] + 1)
        return B

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x)
        y1 = bb.emit_te(te_func, y)
        z1 = bb.emit_te(te_func, z)
        bb.emit_func_output(z1)

    mod = bb.get()
    rx_func = mod["rx_func"]

    prim_func = []
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], PrimFunc):
            prim_func.append(mod[gv])

    # only two PrimFuncs were generated since two of them are equal so got deduped
    assert len(prim_func) == 2
    assert rx_func.body.blocks[0].bindings[0].value.args[0].name_hint == "te_func"
    assert rx_func.body.blocks[0].bindings[1].value.args[0].name_hint == "te_func"
    assert rx_func.body.blocks[0].bindings[2].value.args[0].name_hint == "te_func1"


def test_emit_te_multiple_output():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))

    def te_func(A):
        B0, B1 = te.compute((n, m), lambda i, j: (A[i, j] + 1, A[i, j] * 2), name="B")
        return (B0, B1)

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        z = rx.TupleGetItem(y, 0)
        bb.emit_func_output([y, z])

    rx_func = bb.get()["rx_func"]

    # check call tir output shape is a Tuple of ShapeExpr
    assert rx_func.params[0] == x
    call_node = rx_func.body.blocks[0].bindings[0].value
    assert call_node.op == relay.op.get("relax.call_tir")
    assert call_node.args[0].name_hint == "te_func"
    assert isinstance(call_node.args[2], rx.Tuple)
    assert len(call_node.args[2]) == 2
    assert isinstance(call_node.args[2][0], rx.ShapeExpr)
    assert isinstance(call_node.args[2][1], rx.ShapeExpr)


def test_emit_te_extern():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    y = rx.Var("y", R.Tensor([m, n], "float32"))

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.get()
    rx_func = mod["rx_cblas_matmul"]

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert len(rx_func.body.blocks) == 1
    call_node = rx_func.body.blocks[0].bindings[0].value
    assert isinstance(call_node, rx.Call)
    assert call_node.op == relay.op.get("relax.call_tir")
    assert len(call_node.args) == 3
    assert call_node.args[0].name_hint == "matmul"
    assert call_node.args[1][0] == x
    assert call_node.args[1][1] == y
    assert call_node.args[2][0] == n
    assert call_node.args[2][1] == n


def test_emit_tuple_get_item():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    with bb.function("rx_func"):
        data = nn.Placeholder((n, m, 224, 224), name="x")
        gamma = nn.Parameter((m,))
        beta = nn.Parameter((m,))
        moving_mean = nn.Parameter((m,))
        moving_var = nn.Parameter((m,))
        y = bb.emit_te(topi.nn.batch_norm, data, gamma, beta, moving_mean, moving_var)

        z = bb.emit(rx.TupleGetItem(y, 0))
        assert z.shape[0] == n
        assert z.shape[1] == m
        assert z.shape[2] == 224
        assert z.shape[3] == 224
        assert z.checked_type.ndim == 4
        assert z.checked_type.dtype == "float32"

        w = bb.emit(rx.TupleGetItem(y, 1))
        assert w.shape[0] == m
        assert w.checked_type.dtype == "float32"

        o = bb.emit(rx.TupleGetItem(y, 2))
        assert o.shape[0] == m
        assert o.checked_type.dtype == "float32"
        bb.emit_func_output([y, w], params=[data, gamma, beta, moving_mean, moving_var])

    func = bb.get()["rx_func"]
    assert len(func.body.blocks[0].bindings) == 4


def test_nested_function_fail():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, x))
            with bb.function("func1", [x, y]):
                gv1 = bb.emit(rx.op.add(x, x))
            bb.emit_func_output(gv0)


def test_emit_func_output_twice_fail():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0)
            bb.emit_func_output(gv0)


def test_func_params_twice_fail():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0, [x])


def test_no_func_params_fail():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func"):
            gv0 = bb.emit(rx.Call(ExternFunc("test.blockbuilder.nop"), []))
            bb.emit_func_output(gv0)


def test_block_builder_scope_recovery():
    bb = rx.BlockBuilder()

    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = rx.Var("x", R.Tensor([n, m], "float32"))
    y = rx.Var("y", R.Tensor([m, n], "float32"))

    with pytest.raises(RuntimeError):
        # this line fails
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))

    # current should be recovered
    assert rx.BlockBuilder.current() is None

    # second attempt to do it correctly.
    with bb.function("func", [x, y]):
        gv0 = bb.emit(rx.op.add(x, y))
        bb.emit_func_output(gv0)


if __name__ == "__main__":
    pytest.main([__file__])
