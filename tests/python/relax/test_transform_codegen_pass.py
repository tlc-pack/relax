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

from __future__ import annotations
import pytest

import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.script import relax as R
from tvm import transform

env_checker_codegen = tvm.get_global_func("relax.ext.tensorrt", True)
env_checker_runtime = tvm.get_global_func("relax.op.is_tensorrt_runtime_enabled", True)

has_tensorrt_codegen = pytest.mark.skipif(
    not env_checker_codegen,
    reason="TensorRT codegen not available",
)
has_tensorrt_runtime = pytest.mark.skipif(
    not env_checker_runtime or not env_checker_runtime(),
    reason="TensorRT runtime not available",
)

# Global variable in pytest that applies markers to all tests.
pytestmark = [has_tensorrt_codegen, has_tensorrt_runtime]


def test_single_annot_func():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            z1 = relax.add(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            lv0 = relax_func(x, y)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    new_relax_mod = mod["relax_func"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_func")
    mod["relax_func"] = new_relax_mod

    # Run Codegen pass
    seq = transform.Sequential(
        [relax.transform.RunCodegen(), relax.transform.RemoveUnusedFunctions()]
    )
    new_mod = seq(mod)

    target_str = "cuda"
    target = tvm.target.Target(target_str)
    dev = tvm.device(target_str, 0)

    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    vm0 = relax.VirtualMachine(ex0, dev)
    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    # Measure the performance w/o tuning log
    out0 = vm0["main"](data0, data1)

    # Correct output: Current relax cannot lower relax.add.
    # Use numpy baseline instead.
    tmp = np0 + np1
    out1 = tmp + tmp
    out1 = out1 + tmp
    tvm.testing.assert_allclose(out0.numpy(), out1)

    # If the annotation does not match with the target codegen, do not perform the codegen process.
    new_mod = relax.transform.RunCodegen(target_codegens=["INVALID_CODEGEN"])(mod)
    tvm.ir.assert_structural_equal(mod, new_mod)


# TODO(@sunggg):  test with more complex patterns (e.g., multiple annots, mixed codegens, different ops, const binding)

if __name__ == "__main__":
    pytest.main([__file__])
