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


@tvm.register_func("test.vm.move")
def move(src):
    return src

@tvm.register_func("test.vm.add")
def add(a, b):
    ret = a.asnumpy() + b.asnumpy()
    return tvm.nd.array(ret)

@tvm.register_func("test.vm.mul")
def mul(a, b):
    ret = a.asnumpy() * b.asnumpy()
    return tvm.nd.array(ret)

def test_vm_execute():
    ib = rx.Builder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = rx.VirtualMachine(ex)
    a = tvm.nd.array(np.random.rand(4,))
    b = tvm.nd.array(np.random.rand(4,))
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())

def test_vm_multiple_func():
    ib = rx.Builder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = rx.VirtualMachine(ex)
    a = tvm.nd.array(np.random.rand(4,))
    b = tvm.nd.array(np.random.rand(4,))
    mul_res = vm["func1"](a, b)
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())
    np.testing.assert_allclose(mul_res.asnumpy(), a.asnumpy() * b.asnumpy())


def test_vm_serialize():
    ib = rx.Builder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    exec0 = ib.get()
    exec0.save_to_file("exec.bin")
    exec1 = rx.load_exec_from_file("exec.bin")
    assert exec0.astext() == exec1.astext()

if __name__ == "__main__":
    test_vm_execute()
    test_vm_multiple_func()
    test_vm_serialize()
