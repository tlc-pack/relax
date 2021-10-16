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
from tvm import relay
from tvm import relax as rx


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

    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            assert lv1.name_hint == "lv1"
            gv0 = ib.emit_output(lv1)
        assert gv0.name_hint == "gv"
        ib.emit_func_output(gv0)

    func = ib.get()
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

    with ib.function([x, y], "func"):
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

    func = ib.get()
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

    with ib.function([x, y, z, w]):
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

    with ib.function([x, y]):
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
    func = ib.get()
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


if __name__ == "__main__":
    test_block_builder()
    test_function_single_block()
    test_function_multi_blocks()
    test_binary_shape_type_deduction()
    test_emit_match_shape()
    test_normalize()
