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

@tvm.register_func("vm.builtin.print")
def vmprint(obj):
    print(obj)

@tvm.register_func("vm.builtin.move")
def move(src):
    return src

@tvm.register_func("vm.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)

ib = rx.Builder()
print(ib)

arr = tvm.nd.array(np.random.rand(4,))

ib.emit_call("vm.builtin.move", args=[arr], ret=ib.r(0))
ib.emit_call("vm.builtin.move", args=[ib.r(0)], ret=ib.r(1))
ib.emit_call("vm.add", args=[ib.r(0), ib.r(1)], ret=ib.r(2))
ib.emit_call("vm.builtin.print", args=[ib.r(2)])

exec0 = ib.get()
print(exec0)
print(exec0.stats())
print(exec0.astext())

exec0.save_to_file("exec.bin")
exec1 = rx.load_exec_from_file("exec.bin")
print(exec1)
print(exec1.stats())
print(exec1.astext())

vm = rx.VirtualMachine(exec0)
print(vm)
vm.run()
print("original: ")
print(arr)
exit(0)
