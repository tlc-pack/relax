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
import numpy as np
import pytest
import tvm
from tvm import relax as rx
from tvm import tir
from tvm.script import relax as R


def test_var() -> None:
    v0 = rx.Var("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0._checked_type_ is None
    assert v0.struct_info_ is None
    shape = [54, 96]
    v1 = rx.Var("v1", R.Tensor(shape, "float32"))
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape):
        assert s0 == s1
    assert v1.checked_type == rx.DynTensorType(2, "float32")
    tvm.ir.assert_structural_equal(v1.struct_info, rx.TensorStructInfo(shape, "float32"))


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    assert v0.name_hint == "v0"
    assert v0.shape_ is None
    assert v0._checked_type_ is None
    assert v0.struct_info_ is None

    shape = [54, 96]
    v1 = rx.DataflowVar("v1", R.Tensor(shape, "float16"))
    assert v1.name_hint == "v1"
    for s0, s1 in zip(v1.shape_, shape):
        assert s0 == s1
    assert v1._checked_type_ == rx.DynTensorType(2, "float16")
    assert isinstance(v1, rx.DataflowVar)
    tvm.ir.assert_structural_equal(v1.struct_info, rx.TensorStructInfo(shape, "float16"))


def test_match_shape() -> None:
    # match_shape([16, 8], [m, n])
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", R.Shape())
    b0 = rx.MatchShape(shape, [m, n], var)
    assert b0.value == shape
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.var is not None
    assert b0.var.checked_type == rx.ShapeType()

    # var1: R.Tensor((m, n), "float32") =
    #   match_shape(var0: R.Tensor("float32", ndim=-1), [m, n])
    value = rx.Var("value", R.Tensor("float32", ndim=-1))

    var = rx.Var("v1", R.Tensor([m, n], "float32"))
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


def test_func():
    x = rx.Var("foo", R.Tensor("float32", ndim=2))
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]

    seqe = rx.SeqExpr(blocks, x)
    ret_struct_info = R.Tensor("float32", ndim=-1)
    func = rx.Function([x], seqe, ret_struct_info)
    func = func.with_attr("global_symbol", "func")
    assert func.params[0] == x
    assert func.body == seqe
    assert func.ret_struct_info == ret_struct_info
    assert func.attrs["global_symbol"] == "func"


def test_shape_of():
    v0 = rx.Var("v0")
    s0 = v0.shape
    assert isinstance(s0, rx.Call)
    assert s0.op.name == "relax.shape_of"

    shape = [96, 54]
    v1 = rx.Var("v1", R.Tensor(shape))
    s1 = v1.shape
    for x, y in zip(shape, s1):
        assert x == y


def test_shape_expr():
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    s = rx.ShapeExpr([m, n])
    assert s.values[0] == m
    assert s.values[1] == n
    assert isinstance(s.struct_info, rx.ShapeStructInfo)

    shape_expr = rx.ShapeExpr([10, 20])
    assert shape_expr.values[0] == 10
    assert shape_expr.values[1] == 20
    assert shape_expr.checked_type == rx.ShapeType(ndim=2)
    assert shape_expr.shape_ is None

    x = rx.Var("v0", R.Tensor((10, 20), "float32"))
    assert x.shape_.values[0] == 10
    assert x.shape_.values[1] == 20
    assert x.shape_.checked_type == rx.ShapeType(ndim=2)
    assert x.shape_.shape_ is None

    m = tir.Var("m", "int32")
    with pytest.raises(
        tvm.TVMError, match="the value in ShapeStructInfo can only have dtype of int64"
    ):
        rx.ShapeExpr([m, 3])


if __name__ == "__main__":
    pytest.main([__file__])
