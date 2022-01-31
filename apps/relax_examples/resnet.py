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

# Example ResNet workload by converting the Relay graph to Relax

import tvm
from tvm.relay import testing
from tvm import relax
from tvm.relax.testing import relay_translator, nn
from tvm.script import relax as R
import numpy as np
import time
from tvm import relay
import tvm.runtime.vm as vm_rt


if __name__ == "__main__":
    net, params = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")

    bb = relax.BlockBuilder()
    with bb.function("main"):
        relay_translator.from_relay(net["main"])

    mod = bb.get()
    # print(R.parser.astext(mod["main"]))

    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
    # print(ex.astext())
    vm = relax.VirtualMachine(ex, tvm.cpu(0), mod=lib)

    shape = (1, 3, 224, 224)
    data = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    params = nn.init_params(mod)

    # start_time = time.time()
    # for i in range(10):
    #     print("round", i)
    #     vm["main"](data, *params)
    # end_time = time.time()
    # tvm_time = end_time - start_time
    # print("Relax time elapsed", tvm_time)

    # # print(net)
    exe = relay.vm.compile(net, target=target)
    # # print(exe.bytecode)
    # vm = vm_rt.VirtualMachine(exe, tvm.cpu(0))

    # executor = relay.build_module.create_executor("vm", net, tvm.cpu(0), target="llvm").evaluate()
    # start_time = time.time()
    # for i in range(1):
    #     print("round", i)
    #     # executor(data, *params)
    #     vm.run(data, *params)
    # end_time = time.time()
    # tvm_time = end_time - start_time
    # print("Relay time elapsed", tvm_time)