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
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    assert relax.op.negative(x).op == Op.get("relax.negative")
    assert relax.op.sin(x).op == Op.get("relax.sin")
    assert relax.op.cos(x).op == Op.get("relax.cos")
    assert relax.op.tanh(x).op == Op.get("relax.tanh")
    assert relax.op.sqrt(x).op == Op.get("relax.sqrt")
    assert relax.op.log(x).op == Op.get("relax.log")
    assert relax.op.sigmoid(x).op == Op.get("relax.sigmoid")
    assert relax.op.unique(x).op == Op.get("relax.unique")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_unary_arith_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.negative(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.sin(x1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.log(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.sigmoid(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.tanh(x4), relax.TensorStructInfo(dtype=""))


def test_unary_arith_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.sqrt(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.sigmoid(x1), relax.TensorStructInfo((4, n), "float32"))


def test_unary_arith_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.log(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.tanh(x1), relax.TensorStructInfo(s1, "float32"))


def test_unary_arith_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.negative(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, relax.op.negative(x1), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.negative(x2), relax.TensorStructInfo((2, 3), "int64"))


def test_unary_arith_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.sin(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.log(x1))


def test_unary_arith_wrong_input_number():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        relax.op.cos(x, x)
    with pytest.raises(TypeError):
        relax.op.tanh(x, x, x)


def test_unary_arith_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.cos(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tanh(x1))


if __name__ == "__main__":
    tvm.testing.main()
