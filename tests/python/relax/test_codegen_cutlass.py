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
import numpy as np
import tvm
import tvm.testing

from tvm import relax
from tvm.script import relax as R
from tvm.relax.dpl import make_fused_bias_activation_pattern
from tvm.contrib.cutlass.build import finalize_modules_relax
from tvm.relax.transform import LegalizeOps


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


def get_ref(data_np, weight_np, bias_np):
    mod = LegalizeOps()(Conv2dBiasReLU)
    target = "llvm"
    dev = tvm.device(target, 0)
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)

    data = tvm.nd.array(data_np, dev)
    weight = tvm.nd.array(weight_np, dev)
    bias = tvm.nd.array(bias_np, dev)

    return vm["main"](data, weight, bias).numpy()


has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

cutlass_enabled = pytest.mark.skipif(
    not has_cutlass,
    reason="CUTLASS note enabled.",
)

pytestmark = [cutlass_enabled]


def test_conv2d_offload():
    data_np = np.random.randn(16, 32, 32, 16).astype("float16")
    weight_np = np.random.randn(32, 3, 3, 16).astype("float16")
    bias_np = np.random.randn(1, 1, 1, 32).astype("float16")

    pat = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu"
    )

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(
                [("cutlass.conv2d_bias_relu", pat)], annotate_codegen=True
            ),
            relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}}),
        ]
    )

    mod = seq(Conv2dBiasReLU)

    target = tvm.target.Target("cuda")
    ex = relax.vm.build(mod, target)
    ex = finalize_modules_relax(ex)

    dev = tvm.gpu(0)
    vm = relax.VirtualMachine(ex, dev)

    data = tvm.nd.array(data_np, dev)
    weight = tvm.nd.array(weight_np, dev)
    bias = tvm.nd.array(bias_np, dev)
    out = vm["main"](data, weight, bias).numpy()

    ref = get_ref(data_np, weight_np, bias_np)
    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_conv2d_offload()
