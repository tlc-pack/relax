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
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    assert relax.op.nn.max_pool2d(x).op == Op.get("relax.nn.max_pool2d")
    assert relax.op.nn.adaptive_avg_pool2d(x).op == Op.get("relax.nn.adaptive_avg_pool2d")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_max_pool2d_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))

    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, pool_size=(5, 3)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x0, padding=1), relax.TensorStructInfo((2, 3, 34, 34), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, padding=[1, 2]),
        relax.TensorStructInfo((2, 3, 34, 36), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x6, layout="NCHW16c", out_layout="NHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.nn.max_pool2d(x4), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.nn.max_pool2d(x5), relax.TensorStructInfo(dtype="", ndim=4))


def test_max_pool2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    kh = tir.Var("kh", "int64")
    kw = tir.Var("kw", "int64")
    stride_h = tir.Var("stride_h", "int64")
    stride_w = tir.Var("stride_w", "int64")
    padding_t = tir.Var("padding_t", "int64")
    padding_l = tir.Var("padding_l", "int64")
    padding_b = tir.Var("padding_b", "int64")
    padding_r = tir.Var("padding_r", "int64")
    dilation_h = tir.Var("dilation_h", "int64")
    dilation_w = tir.Var("dilation_w", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool2d(
            x0,
            pool_size=(kh, kw),
            strides=(stride_h, stride_w),
            padding=(padding_t, padding_l, padding_b, padding_r),
            dilation=(dilation_h, dilation_w),
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.div(ih + padding_t + padding_b - dilation_h * (kh - 1) - 1, stride_h) + 1,
                tvm.tir.div(iw + padding_l + padding_r - dilation_w * (kw - 1) - 1, stride_w) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NCHW16c", out_layout="NHWC"),
        relax.TensorStructInfo((n, ih, iw, c * 16), "float32"),
    )


def test_max_pool2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_max_pool2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float16")
    )
    _check_inference(bb, relax.op.nn.max_pool2d(x1), relax.TensorStructInfo((2, 3, 32, 32), "int8"))
    _check_inference(
        bb, relax.op.nn.max_pool2d(x2), relax.TensorStructInfo((2, 3, 32, 32), "int64")
    )


def test_max_pool2d_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, strides=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, dilation=(1, 2, 3))


def test_max_pool2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x, out_layout="OHWI"))


def test_max_pool2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x1))


def test_max_pool2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x1))


def test_adaptive_avg_pool2d_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=30),
        relax.TensorStructInfo((2, 3, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=(28, 30)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x6, layout="NCHW16c", out_layout="NHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x4), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x5), relax.TensorStructInfo(dtype="", ndim=4)
    )


def test_adaptive_avg_pool2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    oh = tir.Var("oh", "int64")
    ow = tir.Var("ow", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((n, c, ih, iw), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=oh),
        relax.TensorStructInfo((n, c, oh, oh), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=(oh, ow)),
        relax.TensorStructInfo((n, c, oh, ow), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NCHW16c", out_layout="NHWC"),
        relax.TensorStructInfo((n, ih, iw, c * 16), "float32"),
    )


def test_adaptive_avg_pool2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=32),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NCHW16c"),
        relax.TensorStructInfo(s1, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x2, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


def test_adaptive_avg_pool2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float16")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x1), relax.TensorStructInfo((2, 3, 32, 32), "int8")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x2), relax.TensorStructInfo((2, 3, 32, 32), "int64")
    )


def test_adaptive_avg_pool2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x, out_layout="OHWI"))


def test_adaptive_avg_pool2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x1))


def test_adaptive_avg_pool2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x1))


if __name__ == "__main__":
    tvm.testing.main()
