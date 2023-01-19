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
from tvm.script import relax as R
from tvm.relax.dpl.pattern import make_fused_bias_activation_pattern


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


@tvm.script.ir_module
class Conv2dConv2dReLU:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.conv2d(data, weight1, padding=(1, 1))
            conv2d = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2d)

        return conv2d


@tvm.script.ir_module
class Conv2dReLUx2Partitioned:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = fused_relax_nn_conv2d_relax_nn_relu(
                data, weight1
            )
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = fused_relax_nn_conv2d_relax_nn_relu1(
                lv, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu1(
        conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv1, weight21, padding=[0, 0, 0, 0]
            )
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv2)
            R.output(gv2)
        return gv2


@tvm.script.ir_module
class Conv2dConv2dReLUPartitioned:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = fused_relax_nn_conv2d(data, weight1)
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = fused_relax_nn_conv2d_relax_nn_relu(
                lv, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(
        conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv1, weight21, padding=[0, 0, 0, 0]
            )
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv2: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            R.output(gv2)
        return gv2


@tvm.script.ir_module
class Conv2dReLUx2Partitioned_only_conv2d:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = fused_relax_nn_conv2d(data, weight1)
            conv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            lv1: R.Tensor((1, 64, 54, 54), dtype="float32") = fused_relax_nn_conv2d1(conv1, weight2)
            conv2d: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(conv2d)
        return conv2d

    @R.function
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d1(
        conv11: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv11, weight21, padding=[0, 0, 0, 0]
            )
            R.output(gv1)
        return gv1


conv2d_pat = make_fused_bias_activation_pattern("relax.nn.conv2d", activation=None)
conv2d_relu_pat = make_fused_bias_activation_pattern("relax.nn.conv2d", activation="relax.nn.relu")


def test_partition_conv2d_relu():
    partitioned = relax.transform.FuseOpsByPattern([("dnnl.conv2d_relu", conv2d_relu_pat)])(
        Conv2dReLUx2
    )
    tvm.ir.assert_structural_equal(partitioned, Conv2dReLUx2Partitioned)


def test_partition_multiple_patterns():
    partitioned = relax.transform.FuseOpsByPattern(
        [("dnnl.conv2d_relu", conv2d_relu_pat), ("dnnl.conv2d", conv2d_pat)]
    )(Conv2dConv2dReLU)

    tvm.ir.assert_structural_equal(partitioned, Conv2dConv2dReLUPartitioned)


def test_partition_order():
    partitioned = relax.transform.FuseOpsByPattern(
        [("dnnl.conv2d", conv2d_pat), ("dnnl.conv2d_relu", conv2d_relu_pat)]
    )(Conv2dReLUx2)

    tvm.ir.assert_structural_equal(partitioned, Conv2dReLUx2Partitioned_only_conv2d)


if __name__ == "__main__":
    pytest.main([__file__])
