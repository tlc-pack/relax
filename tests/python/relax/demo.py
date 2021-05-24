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
from tvm import relax as rx

@tvm.register_func("vm.func0")
def func0():
    print("lalal0")

@tvm.register_func("vm.func1")
def func1():
    print("lalal1")

ib = rx.Builder()
print(ib)

arr = tvm.nd.array(np.random.rand(32, 32))

ib.emit_call("vm.func0", args=[ib.r(0), ib.r(1)])
ib.emit_call("vm.func1", args=[ib.r(2), ib.imm(7482)], ret=ib.r(5))
ib.emit_call("vm.func1", args=[ib.r(3), arr])

executable = ib.get()
print(executable)
print(executable.astext())

executable.save_to_file("exec.bin")
executable1 = rx.load_exec_from_file("exec.bin")
print(executable1)
print(executable1.astext())
exit(0)

vm = rx.VirtualMachine()
vm.load(executable)
vm.execute()

