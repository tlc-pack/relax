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

# Example code on creating, compiling, and running a neural network with pytorch-like API


import tvm
from tvm.relay import Call
from tvm import relax, tir
from tvm.relax.testing import nn
from tvm.script import relax as R
import numpy as np


if __name__ == "__main__":
    builder = nn.Builder()
    n = tir.Var("n", "int64")

    input_size = n
    hidden_sizes = [128, 32]
    output_size = 10

    # build a three linear-layer neural network for a classification task
    with builder.func("main"):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(),
        )
        data = nn.Placeholder((n, 1), name="data")
        result = model(data)
        builder.finalize([data] + model.parameters(), result)

    # get and print the IRmodule being built
    mod = builder.get()
    print(R.parser.astext(mod))

    # build and create vm executor
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = relax.vm.build(mod, target, target_host)
    vm = relax.VirtualMachine(ex, tvm.cpu(), mod=lib)

    # init parameters
    # TODO(@yuchen): implment init_params() to initialize parameters
    n = 784
    linear_weight0 = tvm.nd.array(np.random.rand(n, 128).astype(np.float32))
    linear_bias0 = tvm.nd.array(np.random.rand(128,).astype(np.float32))
    linear_weight1 = tvm.nd.array(np.random.rand(hidden_sizes[0], hidden_sizes[1]).astype(np.float32))
    linear_bias1 = tvm.nd.array(np.random.rand(hidden_sizes[1],).astype(np.float32))
    linear_weight2 = tvm.nd.array(np.random.rand(hidden_sizes[1], output_size).astype(np.float32))
    linear_bias2 = tvm.nd.array(np.random.rand(output_size,).astype(np.float32))
    params = [linear_weight0, linear_bias0, linear_weight1, linear_bias1, linear_weight2, linear_bias2]

    data = tvm.nd.array(np.random.rand(n, 1).astype(np.float32))
    res = vm["main"](data, *params)
    print(res)
