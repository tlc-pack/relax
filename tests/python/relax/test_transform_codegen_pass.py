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
import os
import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.script import relax as R
from tvm import transform

env_checker_codegen = tvm.get_global_func("relax.ext.tensorrt", True)
env_checker_runtime = tvm.get_global_func("relax.is_tensorrt_runtime_enabled", True)

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


def check_executable(exec, dev, inputs, expected):
    vm = relax.VirtualMachine(exec, dev)
    # Measure the performance w/o tuning log
    out = vm["main"](*inputs)
    tvm.testing.assert_allclose(out.numpy(), expected)


# TODO(sunggg): Serialize TRT runtime module. This might be helpful: `module.export_library(file_name)``
def check_roundtrip(exec0, dev, inputs, expected):
    exec0.mod.export_library("exec.so")
    exec1 = relax.vm.Executable(tvm.runtime.load_module("exec.so"))
    os.remove("exec.so")
    assert exec0.stats() == exec1.stats()
    assert exec0.as_text() == exec1.as_text()

    check_executable(exec0, dev, inputs, expected)
    check_executable(exec1, dev, inputs, expected)


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
    new_relax_func = mod["relax_func"].with_attr("Codegen", "tensorrt")
    new_relax_func = new_relax_func.with_attr("global_symbol", "trt_relax_func")
    mod["relax_func"] = new_relax_func

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

    # Correct output: Current relax cannot lower relax.add.
    # Use numpy baseline instead.
    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    tmp = np0 + np1
    out1 = tmp + tmp
    expected = out1 + tmp

    check_executable(ex0, dev, [data0, data1], expected)
    # TODO: Check if serialization works and the correctness of both original and deserialized execs.
    # check_roundtrip(ex0, dev, [data0, data1], expected)

    # If the annotation does not match with the target codegen, do not perform the codegen process.
    new_mod = relax.transform.RunCodegen(target_codegens=["INVALID_CODEGEN"])(mod)
    tvm.ir.assert_structural_equal(mod, new_mod)


# TODO(@sunggg):  test with more complex patterns (e.g., multiple annots, mixed codegens, different ops, const binding)

if __name__ == "__main__":
    pytest.main([__file__])
