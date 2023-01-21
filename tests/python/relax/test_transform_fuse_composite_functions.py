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
from tvm.relax.dpl.pattern import make_fused_bias_activation_pattern, wildcard, is_op


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.relu(R.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = R.nn.relu(R.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


@tvm.script.ir_module
class Branch:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight)
            relu1 = R.nn.relu(conv1)
            gelu1 = R.nn.gelu(conv1)

            out = R.add(relu1, gelu1)
            R.output(out)

        return out


@tvm.script.ir_module
class BranchMerge:
    @R.function
    def main(
        x1: R.Tensor((10,), "float32"),
        x2: R.Tensor((10,), "float32"),
    ):
        with R.dataflow():
            relu1 = R.nn.relu(x1)
            gelu1 = R.nn.gelu(x2)

            out = R.add(relu1, gelu1)
            R.output(out)

        return out


@tvm.script.ir_module
class MergeCompilerRegionsExample:
    @R.function
    def main(
        x1: R.Tensor((10,), "float32"),
        x2: R.Tensor((10,), "float32"),
        x3: R.Tensor((10,), "float32"),
    ):
        with R.dataflow():
            add1 = R.add(x1, x2)
            gelu1 = R.nn.gelu(x3)
            add2 = R.add(add1, gelu1)
            gelu2 = R.nn.gelu(add2)
            relu1 = R.nn.relu(add2)
            add3 = R.add(gelu2, relu1)
            relu2 = R.nn.relu(add3)
            R.output(relu2)

        return relu2


@tvm.script.ir_module
class MergeCompilerRegionsExampleRef:
    @R.function
    def fused_fused_relax_add_fused_relax_nn_relu(lv1: R.Tensor((10,), dtype="float32"), lv: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"Primitive": 1, "Codegen": "compiler_A", "global_symbol": "fused_fused_relax_add_fused_relax_nn_relu"})
        # block 0
        with R.dataflow():
            @R.function
            def lv2(x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
                # function attr dict
                R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
                # block 0
                with R.dataflow():
                    gv: R.Tensor((10,), dtype="float32") = R.add(x1, x2)
                    R.output(gv)
                return gv

            lv21: R.Tensor((10,), dtype="float32") = lv2(lv1, lv)
            @R.function
            def lv11(add2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
                # function attr dict
                R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
                # block 0
                with R.dataflow():
                    gv1: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
                    R.output(gv1)
                return gv1

            gv2: R.Tensor((10,), dtype="float32") = lv11(lv21)
            R.output(gv2)
        return gv2

    @R.function
    def fused_fused_relax_add_fused_relax_add_fused_relax_nn_relu(x11: R.Tensor((10,), dtype="float32"), x21: R.Tensor((10,), dtype="float32"), lv3: R.Tensor((10,), dtype="float32")) -> R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32")):
        # function attr dict
        R.func_attr({"Primitive": 1, "Codegen": "compiler_A", "global_symbol": "fused_fused_relax_add_fused_relax_add_fused_relax_nn_relu"})
        # block 0
        with R.dataflow():
            @R.function
            def lv22(x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
                # function attr dict
                R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
                # block 0
                with R.dataflow():
                    gv: R.Tensor((10,), dtype="float32") = R.add(x1, x2)
                    R.output(gv)
                return gv

            lv4: R.Tensor((10,), dtype="float32") = lv22(x11, x21)
            gv3: R.Tensor((10,), dtype="float32") = lv22(lv4, lv3)
            @R.function
            def lv31(add2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
                # function attr dict
                R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
                # block 0
                with R.dataflow():
                    gv1: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
                    R.output(gv1)
                return gv1

            gv11: R.Tensor((10,), dtype="float32") = lv31(gv3)
            R.output(gv3, gv11)
        return (gv3, gv11)

    @R.function
    def fused_fused_relax_nn_gelu(x3: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"Primitive": 1, "Codegen": "compiler_B", "global_symbol": "fused_fused_relax_nn_gelu"})
        # block 0
        with R.dataflow():
            @R.function
            def lv41(x31: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
                # function attr dict
                R.func_attr({"Primitive": 1, "Composite": "compiler_B.gelu"})
                # block 0
                with R.dataflow():
                    gv4: R.Tensor((10,), dtype="float32") = R.nn.gelu(x31)
                    R.output(gv4)
                return gv4

            gv5: R.Tensor((10,), dtype="float32") = lv41(x3)
            R.output(gv5)
        return gv5

    @R.function
    def main(x12: R.Tensor((10,), dtype="float32"), x22: R.Tensor((10,), dtype="float32"), x32: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        # block 0
        with R.dataflow():
            lv5: R.Tensor((10,), dtype="float32") = fused_fused_relax_nn_gelu(x32)
            lv12: R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32")) = fused_fused_relax_add_fused_relax_add_fused_relax_nn_relu(x12, x22, lv5)
            lv23: R.Tensor((10,), dtype="float32") = lv12[0]
            lv32: R.Tensor((10,), dtype="float32") = lv12[1]
            lv42: R.Tensor((10,), dtype="float32") = fused_fused_relax_nn_gelu(lv23)
            gv6: R.Tensor((10,), dtype="float32") = fused_fused_relax_add_fused_relax_nn_relu(lv42, lv32)
            R.output(gv6)
        return gv6


relu_pat = is_op("relax.nn.relu")(wildcard())
gelu_pat = is_op("relax.nn.gelu")(wildcard())
add_pat = is_op("relax.add")(wildcard(), wildcard())


def test_conv2d_relu_x2():
    pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=False, activation="relax.nn.relu"
    )

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern([("dnnl.conv2d_relu", pat)]),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(Conv2dReLUx2).script())


def test_branch():
    conv_pat = make_fused_bias_activation_pattern("relax.nn.conv2d")

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                [
                    ("compiler_A.conv2d", conv_pat),
                    ("compiler_A.relu", relu_pat),
                    ("compiler_A.add", add_pat),
                    ("compiler_B.gelu", gelu_pat),
                ]
            ),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(Branch).script())


def test_branch_merge():
    relu_pat = is_op("relax.nn.relu")(wildcard())
    gelu_pat = is_op("relax.nn.gelu")(wildcard())
    add_pat = is_op("relax.add")(wildcard(), wildcard())

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                [
                    ("compiler_A.relu", relu_pat),
                    ("compiler_A.add", add_pat),
                    ("compiler_A.gelu", gelu_pat),
                ]
            ),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    print(seq(BranchMerge).script())


def test_merge_compiler_regions_example():
    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                [
                    ("compiler_A.relu", relu_pat),
                    ("compiler_A.add", add_pat),
                    ("compiler_B.gelu", gelu_pat),
                ]
            ),
            relax.transform.FuseCompositeFunctions(),
        ]
    )

    partitioned = seq(MergeCompilerRegionsExample)

    tvm.ir.structural_equal(partitioned, MergeCompilerRegionsExampleRef)


if __name__ == "__main__":
    # pytest.main([__file__])
    # test_conv2d_relu_x2()
    # test_branch_merge()
    # test_branch()
    test_merge_compiler_regions_example()
