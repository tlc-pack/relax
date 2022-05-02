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
from tvm import tir
from tvm import relax as rx
import numpy as np


def test_var() -> None:
    v0 = rx.Var("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0._checked_type_ is None
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float32")
    v1 = rx.Var("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1._checked_type_ == type_anno


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0._checked_type_ is None
    shape_anno = [54, 96]
    type_anno = rx.DynTensorType(2, "float16")
    v1 = rx.DataflowVar("v1", shape_anno, type_anno)
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape_anno):
        assert s0 == s1
    assert v1._checked_type_ == type_anno
    assert isinstance(v1, rx.DataflowVar)


def test_match_shape() -> None:
    # match_shape([16, 8], [m, n])
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", type_annotation=rx.ShapeType())
    b0 = rx.MatchShape(shape, [m, n], var)
    assert b0.value == shape
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.var is not None
    assert b0.var.checked_type == rx.ShapeType()

    # var1: Tensor((m, n), "float32") =
    #   match_shape(var0: Tensor(_, "float32"), [m, n])
    type_anno0 = rx.DynTensorType(-1, "float32")
    value = rx.Var("value", type_annotation=type_anno0)

    shape_anno = [m, n]
    type_anno = rx.DynTensorType(2, "float32")
    var = rx.Var("v1", shape_anno, type_anno)
    b1 = rx.MatchShape(value, [m, n], var)
    assert b1.value == value
    assert b1.pattern[0] == m
    assert b1.pattern[1] == n
    assert b1.var is not None
    for s0, s1 in zip(b1.var.shape, [m, n]):
        assert s0 == s1
    assert b1.var.checked_type == rx.DynTensorType(2, "float32")


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    assert b0.var.name_hint == "v0"
    assert b0.value == val


def test_binding_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchShape(shape, [m, n], rx.Var("v0"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1


def test_dataflow_block() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchShape(shape, [m, n], rx.Var("v0"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1
    assert isinstance(block0, rx.DataflowBlock)


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    assert seqe.blocks[0] == blocks[0]
    assert seqe.body == x


def test_shape_expr() -> None:
    m = tir.Var("m", dtype="int32")
    n = tir.Var("n", dtype="int32")
    s = rx.ShapeExpr([m, n])
    assert s.values[0] == m
    assert s.values[1] == n


def test_func():
    type_anno = rx.DynTensorType(2, "float32")
    x = rx.Var("foo", type_annotation=type_anno)
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    ret_type = rx.DynTensorType(-1, "float32")
    func = rx.Function([x], seqe, ret_type)
    func = func.with_attr("global_symbol", "func")
    assert func.params[0] == x
    assert func.body == seqe
    assert func.ret_type == ret_type
    assert func.attrs["global_symbol"] == "func"


def test_shape_of():
    v0 = rx.Var("v0")
    s0 = v0.shape
    assert isinstance(s0, tvm.relay.Call)
    assert s0.op.name == "relax.shape_of"

    shape_anno = [96, 54]
    v1 = rx.Var("v1", shape_anno)
    s1 = v1.shape
    for x, y in zip(shape_anno, s1):
        assert x == y


if __name__ == "__main__":
    pytest.main([__file__])
