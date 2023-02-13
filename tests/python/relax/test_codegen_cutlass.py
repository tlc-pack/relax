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
import math
from typing import List, Tuple

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax, relay
from tvm.contrib.cutlass.build import finalize_modules_relax
from tvm.relax.dpl import make_fused_bias_activation_pattern, make_matmul_pattern
from tvm.script import relax as R


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


def get_relay_conv2d_bias_relu(
    d_shape, w_shape, data_dtype="float16", weight_dtype="float16", out_dtype="float16"
):
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    bias = relay.var("bias", shape=(1, 1, 1, w_shape[0]), dtype=out_dtype)
    return relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_dtype=out_dtype,
        )
        + bias
    )


def get_relay_matmul(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    out_dtype="float16",
):
    x = relay.var("x", shape=x_shape, dtype=x_dtype)
    y = relay.var("y", shape=y_shape, dtype=y_dtype)
    return relay.nn.dense(x, y, out_dtype=out_dtype)


def get_relay_matmul_bias(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    bias = relay.var("bias", shape=(y_shape[0],), dtype=bias_dtype)
    return relay.nn.bias_add(
        get_relay_matmul(
            x_shape,
            y_shape,
            x_dtype,
            y_dtype,
            out_dtype,
        ),
        bias,
    )


def get_relay_matmul_bias_relu(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    return relay.nn.relu(
        get_relay_matmul_bias(
            x_shape,
            y_shape,
            x_dtype,
            y_dtype,
            bias_dtype,
            out_dtype,
        )
    )


def get_relay_matmul_bias_gelu(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    bias_add = get_relay_matmul_bias(x_shape, y_shape, x_dtype, y_dtype, bias_dtype, out_dtype)
    mul = bias_add * relay.const((1.0 / math.sqrt(2.0)), dtype=out_dtype)
    if out_dtype == "float16":
        erf = relay.cast(relay.op.erf(relay.cast(mul, "float32")), "float16")
    else:
        erf = relay.op.erf(mul)
    mul_half = erf * relay.const(0.5, dtype=out_dtype)
    add = mul_half + relay.const(0.5, dtype=out_dtype)
    return add * bias_add


def get_relay_ref(relay_expr, *args):
    relay_mod = tvm.IRModule.from_expr(relay_expr)

    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "HWIO"]})]
        )
        relay_mod = seq(relay_mod)

    return (
        relay.create_executor("graph", mod=relay_mod, device=tvm.gpu(0), target="cuda")
        .evaluate()(*args)
        .numpy()
    )


@tvm.script.ir_module
class Conv2dBiasReLU:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight: R.Tensor((32, 3, 3, 16), "float16"),
        bias: R.Tensor((1, 1, 1, 32), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(
                relax.op.add(
                    relax.op.nn.conv2d(
                        data, weight, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    ),
                    bias,
                )
            )
            R.output(conv1)

        return conv1


has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

cutlass_enabled = pytest.mark.skipif(
    not has_cutlass,
    reason="CUTLASS note enabled.",
)

pytestmark = [cutlass_enabled]


def get_result_with_relax_cutlass_offload(mod, patterns: List[Tuple], *args):
    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True),
            relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}}),
        ]
    )

    mod = seq(mod)

    target = tvm.target.Target("cuda")
    ex = relax.vm.build(mod, target)
    ex = finalize_modules_relax(ex)

    dev = tvm.gpu(0)
    vm = relax.VirtualMachine(ex, dev)

    return vm["main"](*(tvm.nd.array(arg, dev) for arg in args)).numpy()


def test_conv2d_offload():
    data = np.random.randn(16, 32, 32, 16).astype("float16")
    weight = np.random.randn(32, 3, 3, 16).astype("float16")
    bias = np.random.randn(1, 1, 1, 32).astype("float16")

    patterns = [
        (
            "cutlass.conv2d_bias_relu",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu"
            ),
        )
    ]
    out = get_result_with_relax_cutlass_offload(Conv2dBiasReLU, patterns, data, weight, bias)

    ref_relay_expr = get_relay_conv2d_bias_relu(data.shape, weight.shape)
    ref = get_relay_ref(ref_relay_expr, data, weight, bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_matmul_offload():
    x = np.random.randn(32, 64).astype("float16")
    y = np.random.randn(64, 128).astype("float16")

    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((32, 64), "float16"), y: R.Tensor((64, 128), "float16")):
            with R.dataflow():
                result = R.matmul(x, y)
                R.output(result)

            return result

    patterns = [
        (
            "cutlass.matmul",
            make_matmul_pattern(
                with_bias=False,
            ),
        ),
    ]
    out = get_result_with_relax_cutlass_offload(Matmul, patterns, x, y)

    ref_relay_expr = get_relay_matmul(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose())

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_matmul_bias_offload():
    x = np.random.randn(32, 64).astype("float16")
    y = np.random.randn(64, 128).astype("float16")
    bias = np.random.randn(
        128,
    ).astype("float16")

    @tvm.script.ir_module
    class MatmulBias:
        @R.function
        def main(
            x: R.Tensor((32, 64), "float16"),
            y: R.Tensor((64, 128), "float16"),
            bias: R.Tensor((128,), "float16"),
        ):
            with R.dataflow():
                result = R.matmul(x, y) + bias
                R.output(result)

            return result

    patterns = [
        (
            "cutlass.matmul_bias",
            make_matmul_pattern(
                with_bias=True,
            ),
        ),
    ]
    out = get_result_with_relax_cutlass_offload(MatmulBias, patterns, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_matmul_bias_relu_offload():
    x = np.random.randn(32, 64).astype("float16")
    y = np.random.randn(64, 128).astype("float16")
    bias = np.random.randn(
        128,
    ).astype("float16")

    @tvm.script.ir_module
    class MatmulBiasRelu:
        @R.function
        def main(
            x: R.Tensor((32, 64), "float16"),
            y: R.Tensor((64, 128), "float16"),
            bias: R.Tensor((128,), "float16"),
        ):
            with R.dataflow():
                result = R.nn.relu(R.matmul(x, y) + bias)
                R.output(result)

            return result

    patterns = [
        (
            "cutlass.matmul_bias_relu",
            make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
        ),
    ]
    out = get_result_with_relax_cutlass_offload(MatmulBiasRelu, patterns, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias_relu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_matmul_bias_gelu_offload():
    x = np.random.randn(32, 64).astype("float16")
    y = np.random.randn(64, 128).astype("float16")
    bias = np.random.randn(
        128,
    ).astype("float16")

    @tvm.script.ir_module
    class MatmulBiasGelu:
        @R.function
        def main(
            x: R.Tensor((32, 64), "float16"),
            y: R.Tensor((64, 128), "float16"),
            bias: R.Tensor((128,), "float16"),
        ):
            with R.dataflow():
                result = R.nn.gelu(R.matmul(x, y) + bias)
                R.output(result)

            return result

    patterns = [
        (
            "cutlass.matmul_bias_gelu",
            make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
            ),
        ),
    ]
    out = get_result_with_relax_cutlass_offload(MatmulBiasGelu, patterns, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias_gelu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
