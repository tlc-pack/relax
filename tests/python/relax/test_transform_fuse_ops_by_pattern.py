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

import numpy as np
import tvm

from tvm import relax
from tvm.script import relax as R
from tvm.relax.dpl.pattern import make_conv_pattern


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
            conv2d = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2d)

        return conv2d


def test_partition_conv2d_relu():
    pat = make_conv_pattern("relax.nn.conv2d", with_bias=False, activation="relax.nn.relu")

    partitioned = relax.transform.FuseOpsByPattern(["dnnl.conv2d_relu"], [pat])(Conv2dReLUx2)
    print(partitioned.script())


if __name__ == "__main__":
    test_partition_conv2d_relu()
