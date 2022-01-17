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
import pytest
import tvm
from tvm import tir, te
from tvm import relay
from tvm import relax as rx

from tvm.ir.base import assert_structural_equal
from tvm.relax import ExternFunc, ShapeExpr, op


@tvm.register_func("test.blockbuilder.nop")
def nop():
    pass


def test_block_builder():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.BlockBuilder()

    ib._begin_binding_block()
    gv0 = ib.emit(rx.op.add(x, y))
    ib._begin_dataflow_block()
    lv0 = ib.emit(rx.op.multiply(gv0, y))
    gv1 = ib.emit_output(rx.op.multiply(lv0, lv0))
    b0 = ib._end_block()
    ib._begin_dataflow_block()
    lv1 = ib.emit(rx.op.multiply(gv0, y))
    gv2 = ib.emit_output(rx.op.multiply(lv1, lv1))
    b1 = ib._end_block()
    gv3 = ib.emit(rx.op.add(x, y))
    b2 = ib._end_block()

    assert isinstance(b0, rx.DataflowBlock)
    assert isinstance(b1, rx.DataflowBlock)
    assert not isinstance(b2, rx.DataflowBlock)


def test_function_single_block():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.BlockBuilder()

    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            assert lv1.name_hint == "lv1"
            gv0 = ib.emit_output(lv1)
        assert gv0.name_hint == "gv"
        ib.emit_func_output(gv0)

    func = ib.get()["func"]
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv0
    assert gv0.shape[0] == m
    assert gv0.shape[1] == n
    assert gv0.checked_type.rank == 2
    assert gv0.checked_type.dtype == "float16"
    assert len(func.body.blocks) == 1
    assert len(func.body.blocks[0].bindings) == 3


def test_function_multi_blocks():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.BlockBuilder()

    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            gv0 = ib.emit_output(lv0)
        assert gv0.name_hint == "gv"
        gv1 = ib.emit(rx.op.add(gv0, gv0))
        assert gv1.name_hint == "gv1"
        with ib.dataflow() as df:
            lv1 = ib.emit(rx.op.add(gv1, gv1))
            assert lv1.name_hint == "lv1"
            gv2 = ib.emit_output(gv1)
        ib.emit_func_output(gv2)

    func = ib.get()["func"]
    assert gv2.shape[0] == m
    assert gv2.shape[1] == n
    assert gv2.checked_type.rank == 2
    assert gv2.checked_type.dtype == "float16"
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.name.name_hint == "func"
    assert func.body.body == gv2
    assert len(func.body.blocks) == 3
    assert len(func.body.blocks[0].bindings) == 2
    assert len(func.body.blocks[1].bindings) == 1
    assert len(func.body.blocks[2].bindings) == 2


def test_multi_functions():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.BlockBuilder()

    with ib.function("func1", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            gv0 = ib.emit_output(lv0)
        ib.emit_func_output(gv0)

    with ib.function("func2", [x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            # TODO(@yuchen): enable block builder to reset local var unique name map
            assert lv0.name_hint == "lv1"
            gv0 = ib.emit_output(lv0)
        ib.emit_func_output(gv0)

    mod = ib.get()
    func1 = mod["func1"]
    assert func1.params[0] == x
    assert func1.params[1] == y
    assert func1.name.name_hint == "func1"
    func2 = mod["func2"]
    assert func2.params[0] == x
    assert func2.params[1] == y
    assert func2.name.name_hint == "func2"


def test_binary_shape_type_deduction():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    k = tir.Var("k", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, 1], dtype0)
    y = rx.Var("y", [n], dtype1)
    z = rx.Var("z", [5], dtype1)
    w = rx.Var("w", [k], dtype1)
    ib = rx.BlockBuilder()

    with ib.function("func", [x, y, z, w]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.shape[0] == m
            assert lv0.shape[1] == n
            assert isinstance(lv0.checked_type, rx.DynTensorType)
            assert lv0.checked_type.rank == 2
            assert lv0.checked_type.dtype == "float16"

            lv1 = ib.emit(rx.op.multiply(x, z))
            assert lv1.shape[0] == m
            assert lv1.shape[1] == 5
            assert isinstance(lv1.checked_type, rx.DynTensorType)
            assert lv1.checked_type.rank == 2
            assert lv1.checked_type.dtype == "float16"

            lv2 = ib.emit(rx.op.multiply(z, w))
            assert isinstance(lv2.shape, tvm.relay.Call)
            assert isinstance(lv2.checked_type, rx.DynTensorType)
            assert lv2.checked_type.rank == 1
            assert lv2.checked_type.dtype == "float16"

            lv3 = ib.emit(rx.op.multiply(y, w))
            assert isinstance(lv3.shape, tvm.relay.Call)
            assert isinstance(lv3.checked_type, rx.DynTensorType)
            assert lv3.checked_type.rank == 1
            assert lv3.checked_type.dtype == "float16"
            gv0 = ib.emit_output(lv3)
        ib.emit_func_output(gv0)
        assert isinstance(gv0.shape, tvm.relay.Call)
        assert isinstance(gv0.checked_type, rx.DynTensorType)
        assert gv0.checked_type.rank == 1
        assert gv0.checked_type.dtype == "float16"


def test_emit_match_shape():
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    type_anno0 = rx.DynTensorType(-1, "float32")
    x = rx.Var("tensor_value", type_annotation=type_anno0)
    shape_anno = [16, 8]
    y = rx.Var("shape_value", type_annotation=rx.ShapeType(), shape_annotation=shape_anno)
    ib = rx.BlockBuilder()

    with ib.function("func", [x, y]):
        with ib.dataflow() as df:
            # lv0: Tensor[(m, n), "float32"] =
            #   match_shape(x: Tensor[_, "float32"], [m, n])
            lv0 = ib.match_shape(x, [m, n])
            assert isinstance(lv0, rx.DataflowVar)
            assert lv0.shape[0] == m
            assert lv0.shape[1] == n
            assert lv0.checked_type.rank == 2
            assert lv0.checked_type.dtype == "float32"

            # lv1: Shape = match_shape(shape, [m, n])
            lv1 = ib.match_shape(y, [m, n])
            assert lv1.checked_type == rx.ShapeType()
            gv0 = ib.emit_output(lv1)

        ib.emit_func_output(gv0)
    func = ib.get()["func"]
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


def test_normalize():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.BlockBuilder()

    add_call = rx.op.multiply(x, y)
    assert isinstance(add_call.shape, relay.Call)

    ib.normalize(add_call)
    assert isinstance(add_call.shape, rx.ShapeExpr)
    assert add_call.shape[0] == m
    assert add_call.shape[1] == n


def test_emit_te():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", [n, m], type_anno)
    y = rx.Var("y", [n, m], type_anno)
    z = rx.Var("z", [n, m], type_anno)
    
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
        return tvm.te.create_prim_func([A, B, C, out])

    # check TIR structure matches expected
    assert_structural_equal(mod["te_func"].body, get_tir_func().body)

    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert rx_func.params[2] == z
    assert rx_func.name.name_hint == "rx_func"
    assert rx_func.body.body == out
    assert len(rx_func.body.blocks) == 1
    assert len(rx_func.body.blocks[0].bindings) == 1
    assert isinstance(rx_func.body.blocks[0].bindings[0].value, rx.Call)
    assert rx_func.body.blocks[0].bindings[0].value.op == relay.op.get("relax.call_tir")
    assert len(rx_func.body.blocks[0].bindings[0].value.args) == 3
    assert rx_func.body.blocks[0].bindings[0].value.args[1].name_hint == "te_func"
    assert rx_func.body.blocks[0].bindings[0].value.args[2][0] == x
    assert rx_func.body.blocks[0].bindings[0].value.args[2][1] == y
    assert rx_func.body.blocks[0].bindings[0].value.args[2][2] == z


def test_emit_te_multiple():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", [n, m], type_anno)
    y = rx.Var("y", [n, m], type_anno)

    def te_func(A):
        B = te.compute((128, 128), lambda i, j: A[i, j] + 1)
        return B

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x)
        y1 = bb.emit_te(te_func, y)
        bb.emit_func_output(y1)
    
    func = bb.get()["rx_func"]
    assert func.body.blocks[0].bindings[0].value.args[1].name_hint == "te_func"
    assert func.body.blocks[0].bindings[1].value.args[1].name_hint == "te_func1"

def test_emit_te_multiple_output():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", [n, m], type_anno)

    def te_func(A):
        B0, B1 = te.compute((n, m), lambda i, j: (A[i, j] + 1, A[i, j] * 2), name="B")
        return (B0, B1)

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        z = relay.TupleGetItem(y, 0)
        bb.emit_func_output([y, z])

    rx_func = bb.get()["rx_func"]

    # check call tir output shape is a Tuple of ShapeExpr
    assert rx_func.params[0] == x
    assert rx_func.name.name_hint == "rx_func"
    assert rx_func.body.blocks[0].bindings[0].value.op == relay.op.get("relax.call_tir")
    assert rx_func.body.blocks[0].bindings[0].value.args[1].name_hint == "te_func"
    assert isinstance(rx_func.body.blocks[0].bindings[0].value.args[0], relay.Tuple)
    assert len(rx_func.body.blocks[0].bindings[0].value.args[0]) == 2
    assert isinstance(rx_func.body.blocks[0].bindings[0].value.args[0][0], rx.ShapeExpr)
    assert isinstance(rx_func.body.blocks[0].bindings[0].value.args[0][1], rx.ShapeExpr)

def test_emit_te_extern():
    bb = rx.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("x", [n, m], type_anno)
    y = rx.Var("y", [m, n], type_anno)

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)
    
    mod = bb.get()
    rx_func = mod["rx_cblas_matmul"]
    
    # check Relax function calls TIR function with call_tir call
    assert rx_func.params[0] == x
    assert rx_func.params[1] == y
    assert len(rx_func.body.blocks) == 1
    assert isinstance(rx_func.body.blocks[0].bindings[0].value, rx.Call)
    assert rx_func.body.blocks[0].bindings[0].value.op == relay.op.get("relax.call_tir")
    assert len(rx_func.body.blocks[0].bindings[0].value.args) == 3
    assert rx_func.body.blocks[0].bindings[0].value.args[1].name_hint == "matmul"
    assert rx_func.body.blocks[0].bindings[0].value.args[2][0] == x
    assert rx_func.body.blocks[0].bindings[0].value.args[2][1] == y


def test_nested_function_fail():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, x))
            with bb.function("func1", [x, y]):
                gv1 = bb.emit(rx.op.add(x, x))
            bb.emit_func_output(gv0)


def test_emit_func_output_twice_fail():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0)
            bb.emit_func_output(gv0)


def test_func_params_twice_fail():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func", [x, y]):
            gv0 = bb.emit(rx.op.add(x, y))
            bb.emit_func_output(gv0, [x])


def test_no_func_params_fail():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    bb = rx.BlockBuilder()

    with pytest.raises(RuntimeError):
        with bb.function("func"):
            gv0 = bb.emit(rx.Call(ExternFunc("test.blockbuilder.nop"), None))
            bb.emit_func_output(gv0)


if __name__ == "__main__":
    test_block_builder()
    test_function_single_block()
    test_function_multi_blocks()
    test_multi_functions()
    test_binary_shape_type_deduction()
    test_emit_match_shape()
    test_normalize()
    test_emit_te()
    test_emit_te_multiple()
    test_emit_te_multiple_output()
    test_emit_te_extern()
    test_nested_function_fail()
    test_emit_func_output_twice_fail()
    test_func_params_twice_fail()
    test_no_func_params_fail()

