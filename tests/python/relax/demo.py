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

@tvm.register_func("vm.op.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)

@tvm.register_func("vm.op.mul")
def add(a, b):
    ret = a.asnumpy() * b.asnumpy()
    return tvm.nd.array(ret)


# Building

ib = rx.Builder()

func = ib.emit_func("func0", num_inputs=2)
ib.emit_call("vm.op.add", args=[func.input(0), func.input(1)], ret=ib.r(0))
ib.emit_call("vm.builtin.move", args=[ib.r(0)], ret=ib.r(1))
ib.emit_call("vm.builtin.print", args=[ib.r(1)])
ib.emit_ret(ib.r(1))

func = ib.emit_func("func1", num_inputs=2)
ib.emit_call("vm.op.mul", args=[func.input(0), func.input(1)], ret=ib.r(0))
ib.emit_call("vm.builtin.move", args=[ib.r(0)], ret=ib.r(1))
ib.emit_call("vm.builtin.print", args=[ib.r(1)])
ib.emit_ret(ib.r(1))

exec0 = ib.get()
print("============")
print("Executable 0")
print(exec0)
print(exec0.stats())
print(exec0.astext())
# print(exec0.aspython())

# Execution

vm = rx.VirtualMachine(exec0)
print(vm)
a = tvm.nd.array(np.random.rand(4,))
b = tvm.nd.array(np.random.rand(4,))
mul_res = vm["func1"](a, b)
add_res = vm["func0"](a, b)
np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())
np.testing.assert_allclose(mul_res.asnumpy(), a.asnumpy() * b.asnumpy())
exit(0)


# Serialization and Deserialization

# exec0.save_to_file("exec.bin")
# exec1 = rx.load_exec_from_file("exec.bin")
# print("============")
# print("Executable 1")
# print(exec1)
# print(exec1.stats())
# print(exec1.astext())
# print(exec1.aspython())
# print("============")
