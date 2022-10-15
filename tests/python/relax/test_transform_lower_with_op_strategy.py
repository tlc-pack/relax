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

import tempfile

import numpy as np
import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax
from tvm.relax.testing import transform
from tvm.script import relax as R
from tvm.target import Target


@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
    ) -> Tensor((16, 16), "float32"):
        gv0 = R.multiply(x, w)
        gv1 = R.add(x, gv0)
        return gv1


def build_and_run(mod, target, dev, np_inputs):
    inputs = [tvm.nd.array(np_input, dev) for np_input in np_inputs]
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.relax_integration.tune_relax(
            mod=mod,
            params=None,
            target=target,
            work_dir=work_dir,
            num_trials_per_iter=20,
            max_trials_global=20,
            task_scheduler="round-robin",
        )
        ex = ms.relax_integration.compile_relax(db, mod, target, params=None)
    vm = relax.VirtualMachine(ex, dev)
    vm["main"](*inputs)


def _test_lowering(target, dev):
    mod = InputModule
    assert mod
    with tvm.transform.PassContext(opt_level=3):
        out_mod = transform.LowerWithRelayOpStrategyPass(target)(mod)

    input_shape = (16, 16)
    np_inputs = [
        np.random.rand(*input_shape).astype(np.float32),
        np.random.rand(*input_shape).astype(np.float32),
    ]
    build_and_run(out_mod, target, dev, np_inputs)


def test_lowering_cpu(target_str="llvm --num-cores=16"):
    _test_lowering(Target(target_str), tvm.cpu())


@tvm.testing.requires_gpu
def test_lowering_gpu(target_str="nvidia/nvidia-t4"):
    _test_lowering(Target(target_str), tvm.cuda())


if __name__ == "__main__":
    test_lowering_cpu()
    test_lowering_gpu()
