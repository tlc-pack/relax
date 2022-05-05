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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax

from tvm.script import relax as R

import numpy as np


@tvm.script.ir_module
class InputModule:
    @R.function
    def foo(x: Tensor((m, n), "int64")):
        y = relax.unique(x, sorted=False)
        y_sorted = relax.unique(x)
        return y, y_sorted


def test_unique():

    mod = InputModule
    # TODO(prakalp): also add test for compiling and running on cuda device.
    target = tvm.target.Target("llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    data_numpy = np.random.randint(0, 16, (16, 16))
    data = tvm.nd.array(data_numpy)
    result, result_sorted = vm["foo"](data)

    expected_output_sorted, indices = np.unique(data_numpy, return_index=True)
    expected_output = [data_numpy.flatten()[index] for index in sorted(indices, reverse=True)]

    np.testing.assert_array_equal(expected_output_sorted, result_sorted.numpy())
    np.testing.assert_array_equal(expected_output, result.numpy())


if __name__ == "__main__":
    pytest.main([__file__])
