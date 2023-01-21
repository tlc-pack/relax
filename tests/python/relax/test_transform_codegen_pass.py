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
import os
import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.script import relax as R
from tvm.relax.testing import transform
import tempfile
from tvm.relax.transform.tuning_api import Trace

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

# Target gpu
target_str = "nvidia/nvidia-t4"
target = tvm.target.Target(target_str)
dev = tvm.cuda()


def check_executable(exec, dev, inputs, expected):
    vm = relax.VirtualMachine(exec, dev)
    out = vm["main"](*inputs)
    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


def check_roundtrip(exec0, dev, inputs, expected):
    exec0.mod.export_library("exec.so")
    exec1 = relax.vm.Executable(tvm.runtime.load_module("exec.so"))
    os.remove("exec.so")
    assert exec0.stats() == exec1.stats()
    assert exec0.as_text() == exec1.as_text()

    check_executable(exec0, dev, inputs, expected)
    check_executable(exec1, dev, inputs, expected)


def gen_ground_truth(mod, target, dev, inputs):
    # Lower and run tuning
    # Since there is no default schedule for GPU in MS yet, this is necessary
    with tempfile.TemporaryDirectory() as work_dir:
        with target, tvm.transform.PassContext(trace=Trace(mod), opt_level=0):
            seq = tvm.transform.Sequential(
                [
                    transform.LowerWithRelayOpStrategyPass(target),
                    relax.transform.MetaScheduleTuneIRMod(
                        params={}, work_dir=work_dir, max_trials_global=8
                    ),
                    relax.transform.MetaScheduleApplyDatabase(work_dir),
                ]
            )
            new_mod = seq(mod)
    assert relax.analysis.well_formed(new_mod)
    exec = relax.vm.build(new_mod, target, params={})
    vm = relax.VirtualMachine(exec, dev)
    return vm["main"](*inputs)


@tvm.testing.requires_gpu
def test_single_annot_func():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func(
            x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            z1 = relax.multiply(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            lv0: R.Tensor((16, 16), "float32") = relax_func(x, y)
            return lv0

    # Prepare IRModule and its input
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    data0 = tvm.nd.array(np0, dev)
    data1 = tvm.nd.array(np1, dev)
    inputs = [data0, data1]

    # Ground truth should be generated before annotation
    # due to the conflict with MS task extraction
    # TODO(@sunggg): Sort this out
    expected = gen_ground_truth(mod, target, dev, inputs)

    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    new_relax_func = mod["relax_func"].with_attr("Codegen", "tensorrt")
    new_relax_func = new_relax_func.with_attr("global_symbol", "trt_relax_func")
    mod["relax_func"] = new_relax_func

    # Run Codegen pass
    new_mod = relax.transform.RunCodegen()(mod)
    ex0 = relax.vm.build(new_mod, target, params={})

    # Sanity check for the correctness and rountrip
    check_roundtrip(ex0, dev, inputs, expected)

    # If the annotation does not match with the target codegen, do not perform the codegen process.
    new_mod = relax.transform.RunCodegen(target_options={"INVALID_CODEGEN": {}})(mod)
    # TODO(tvm-team): Currently disabled due to the lack of type annotation support during parser.
    #                 Revisit when new version of parser is available.
    # tvm.ir.assert_structural_equal(mod, new_mod)


@tvm.testing.requires_gpu
def test_mix_use_tensorrt_and_tvm():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def byoc_func(
            x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            z1 = relax.multiply(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def tvm_func(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            lv0 = byoc_func(x, y)
            lv1 = tvm_func(x, lv0)
            return lv1

    # Prepare IRModule and its inputs
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    data0 = tvm.nd.array(np0, dev)
    data1 = tvm.nd.array(np1, dev)
    inputs = [data0, data1]
    expected = gen_ground_truth(mod, target, dev, [data0, data1])

    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    new_byoc_func = mod["byoc_func"].with_attr("Codegen", "tensorrt")
    new_byoc_func = new_byoc_func.with_attr("global_symbol", "trt_byoc_func")
    mod["byoc_func"] = new_byoc_func

    # Run Codegen pass
    with tempfile.TemporaryDirectory() as work_dir:
        with target, tvm.transform.PassContext(trace=Trace(mod), opt_level=3):
            seq = tvm.transform.Sequential(
                [
                    relax.transform.RunCodegen(),
                    transform.LowerWithRelayOpStrategyPass(target),
                    relax.transform.MetaScheduleTuneIRMod(
                        params={}, work_dir=work_dir, max_trials_global=8
                    ),
                    relax.transform.MetaScheduleApplyDatabase(work_dir),
                ]
            )
            new_mod = seq(mod)
    assert relax.analysis.well_formed(new_mod)
    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    # Sanity check for the correctness and rountrip
    check_roundtrip(ex0, dev, inputs, expected)


# TODO(@sunggg):  test with more complex patterns (e.g., multiple annots, mixed codegens, different ops, const binding)

if __name__ == "__main__":
    pytest.main([__file__])
