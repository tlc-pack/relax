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
import tvm.testing

from tvm import relax, relay, rpc
from tvm.contrib import utils

from tvm.relax.testing import relay_translator


def get_relay_conv2d_relu_x2(d_shape, w_shape):
    data = relay.var("data", shape=d_shape)
    weight1 = relay.var("weight1", shape=w_shape)
    weight2 = relay.var("weight2", shape=w_shape)
    conv1 = relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight1,
            kernel_size=w_shape[2:],
            padding=(1, 1),
        )
    )
    return relay.nn.relu(
        relay.nn.conv2d(
            data=conv1,
            weight=weight2,
            kernel_size=w_shape[2:],
            padding=(0, 0),
        )
    )


def get_exec(data_shape):
    weight1_np = np.random.randn(64, 64, 3, 3).astype("float32")
    weight2_np = np.random.randn(64, 64, 3, 3).astype("float32")

    relay_mod = tvm.IRModule.from_expr(get_relay_conv2d_relu_x2(data_shape, weight1_np.shape))
    params = {"weight1": weight1_np, "weight2": weight2_np}

    target = "llvm"
    mod = relay_translator.from_relay(relay_mod["main"], target, params)

    return relax.vm.build(mod, target)


def test_conv2d_cpu():
    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    ex = get_exec(data_np.shape)

    vm = relax.VirtualMachine(ex, tvm.cpu(), profile=True)
    report = vm.profile("main", tvm.nd.array(data_np))
    print(report)

    assert "Duration" in str(report)
    assert "conv2d" in str(report)


def test_rpc():
    data_np = np.random.randn(1, 64, 56, 56).astype("float32")
    ex = get_exec(data_np.shape)

    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    ex.mod.export_library(path)

    server = rpc.Server("127.0.0.1")
    remote = rpc.connect(server.host, server.port, session_timeout=10)

    remote.upload(path)
    rexec = remote.load_module("vm_library.so")

    device = remote.cpu()

    vm = relax.vm.VirtualMachine(exec=rexec, device=device, profile=True)
    data = tvm.nd.array(data_np, device)

    vm.profile("main", data)

    vm.set_input("main", data)
    report = vm.profile("main")
    print(report)


if __name__ == "__main__":
    # tvm.testing.main()
    test_rpc()
    # test_conv2d_cpu()
