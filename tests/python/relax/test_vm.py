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
import numpy as np
import tvm
from tvm.relay import Call
from tvm import relax as rx
from tvm.runtime import container


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

@tvm.register_func("test.vm.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.asnumpy())

def test_vm_execute():
    ib = rx.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = rx.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(np.random.rand(4,))
    b = tvm.nd.array(np.random.rand(4,))
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())

def test_vm_multiple_func():
    ib = rx.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = rx.VirtualMachine(ex, tvm.cpu())
    a = tvm.nd.array(np.random.rand(4,))
    b = tvm.nd.array(np.random.rand(4,))
    mul_res = vm["func1"](a, b)
    add_res = vm["func0"](a, b)
    np.testing.assert_allclose(add_res.asnumpy(), a.asnumpy() + b.asnumpy())
    np.testing.assert_allclose(mul_res.asnumpy(), a.asnumpy() * b.asnumpy())

def test_vm_serialize():
    ib = rx.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    arr = tvm.nd.array(np.random.rand(4,))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), arr], dst=ib.r(1))
        ib.emit_ret(ib.r(1))
    exec0 = ib.get()
    exec0.save_to_file("exec.bin")
    exec1 = rx.load_exec_from_file("exec.bin")
    assert exec0.astext() == exec1.astext()

def test_vm_constant_serialize():
    dtype = tvm.DataType('float32')
    shape = (3, 4)
    shape_tuple = container.ShapeTuple(shape)
    input = tvm.nd.array(np.random.rand(3,4).astype(np.float32))
    ib = rx.ExecBuilder()
    with ib.function("main", num_inputs=1):
        ib.emit_call("vm.builtin.alloc_storage", args=[ib.vm_state(), ib.imm(24), ib.imm(64), ib.imm(1), dtype], dst=ib.r(1))
        ib.emit_call("vm.builtin.alloc_tensor", args=[ib.r(1), ib.imm(0), dtype, ib.r(0)], dst=ib.r(2))
        ib.emit_call("test.vm.identity", args=[input, ib.r(2)])
        ib.emit_ret(ib.r(2))
    exec0 = ib.get()
    exec0.save_to_file("exec.bin")
    exec1 = rx.load_exec_from_file("exec.bin")
    assert exec0.astext() == exec1.astext()
    vm = rx.VirtualMachine(exec1, tvm.cpu())
    res = vm["main"](shape_tuple)
    np.testing.assert_allclose(input.asnumpy(), res.asnumpy())

def test_vm_checker():
    ib = rx.ExecBuilder()
    try:
        with ib.function("func0", num_inputs=2):
            ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(2)], dst=ib.r(2))
            ib.emit_ret(ib.r(2))
        ib.get()
    except ValueError as ex:
        assert True

def test_vm_formalize():
    ib0 = rx.ExecBuilder()
    ib1 = rx.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(100))
        ib0.emit_call("test.vm.mul", args=[ib0.r(1), ib0.r(100)], dst=ib0.r(50))
        ib0.emit_ret(ib0.r(50))
    with ib1.function("func0", num_inputs=2):
        ib1.emit_call("test.vm.add", args=[ib1.r(0), ib1.r(1)], dst=ib1.r(2))
        ib1.emit_call("test.vm.mul", args=[ib1.r(1), ib1.r(2)], dst=ib1.r(3))
        ib1.emit_ret(ib1.r(3))
    exec0 = ib0.get()
    exec1 = ib1.get()
    assert exec0.astext() == exec1.astext()

@tvm.register_func("test.vm.add_scalar")
def add_scalar(a, b):
    return a + b

@tvm.register_func("test.vm.get_device_id")
def get_device_id(device):
    return device.device_id

def test_vm_operand():
    ib0 = rx.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add_scalar", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(2))
        ib0.emit_ret(ib0.r(2))
    exec0 = ib0.get()
    vm = rx.VirtualMachine(exec0, tvm.cpu())
    res = vm["func0"](2, 3)
    assert res == 5

    ib1 = rx.ExecBuilder()
    with ib1.function("func1", num_inputs=1):
        ib1.emit_call("test.vm.get_device_id", args=[ib1.r(0)], dst=ib1.r(1))
        ib1.emit_ret(ib1.r(1))
    exec1 = ib1.get()
    vm = rx.VirtualMachine(exec1, tvm.cpu())
    res = vm["func1"](tvm.cpu(3))
    assert res == 3

def test_vm_shapeof():
    ib = rx.ExecBuilder()
    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    with ib.function("main", num_inputs=0):
        ib.emit_call("vm.builtin.shape_of", args=[arr], dst=ib.r(0))
        ib.emit_ret(ib.r(0))
    ex = ib.get()
    vm = rx.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()
    for i, s in enumerate(res):
        assert s == shape[i]


def test_vm_storage():
    ib = rx.ExecBuilder()
    with ib.function("main", num_inputs=7):
        ib.emit_call("vm.builtin.alloc_storage", args=[ib.vm_state(), ib.r(0), ib.r(1), ib.r(2), ib.r(3)], dst=ib.r(7))
        ib.emit_call("vm.builtin.alloc_tensor", args=[ib.r(7), ib.r(4), ib.r(5), ib.r(6)], dst=ib.r(8))
        ib.emit_ret(ib.r(8))
    ex = ib.get()
    vm = rx.VirtualMachine(ex, tvm.cpu())
    dtype = tvm.DataType('float32')
    cpu_dev = tvm.cpu().device_type
    buffer_size = 24
    alignment = 8
    offset = 0
    shape = (32, 16)
    shape_tuple = container.ShapeTuple(shape)
    res = vm["main"](buffer_size, alignment, cpu_dev, dtype, offset, dtype, shape_tuple)
    assert res.device == tvm.cpu()
    assert res.shape == shape

def test_vm_compile_stage0():
    @rx.script
    class Mod:
        def foo(x: Tensor[(3, 4), "float32"]):
            y = relax.call_packed("vm.builtin.alloc_storage", (12,), (64,), device_id=0, device_type=1, attrs_type_key="relax.attrs.AllocStorageAttrs")
            z = relax.call_packed("vm.builtin.alloc_tensor", y, (0,), (3, 4), attrs_type_key="relax.attrs.AllocTensorAttrs")
            w = relax.call_packed("test.vm.identity", x, z)
            return z

    mod = Mod()
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = rx.vm.build(mod, target, target_host)
    inp = tvm.nd.array(np.random.rand(3,4).astype(np.float32))
    vm = rx.VirtualMachine(ex, tvm.cpu(), mod=lib)
    res = vm["foo"](inp)
    np.testing.assert_allclose(inp.asnumpy(), res.asnumpy())


def test_vm_compile_stage1():
    @rx.script
    class Mod1:
        @tvm.script.tir
        def shape_func0(heap: ty.handle) -> None:
            # function attr dict
            tir.func_attr({"global_symbol": "shape_func0"})
            H = tir.match_buffer(heap, [tir.int64(4)], dtype="int64", elem_offset=tir.int64(0), align=128, offset_factor=1)
            # body
            tir.store(H.data, tir.int64(2), (tir.load("int64", H.data, tir.int64(0))*tir.int64(2)), True)
            tir.store(H.data, tir.int64(3), (tir.load("int64", H.data, tir.int64(1))*tir.int64(3)), True)
    
        def foo(x: Tensor[_, "float32"]) -> Shape:
            shape_heap: Tensor[(4,), "int64"] = relax.call_packed("vm.builtin.alloc_shape_heap", (4,))
            gv0 = relax.call_packed("vm.builtin.shape_of", x)
            gv1 = relax.call_packed("vm.builtin.decode_shape", gv0, shape_heap, (0, 1))
            gv2 = shape_func0(shape_heap)
            gv3 = relax.call_packed("vm.builtin.make_shape", shape_heap, (2, 3))
            return gv3

    mod = Mod1()
    code = rx.parser.astext(mod)
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = rx.vm.build(mod, target, target_host)
    vm = rx.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


def test_vm_compile_stage2():
    @rx.script
    class Mod2:
        def foo(x: Tensor[_, "float32"]) -> Shape:
            sh = relax.call_packed("vm.builtin.shape_of", x)
            relax.match_shape(sh, (n, m))
            return (n * 2, m * 3)

    mod = Mod2()
    code = rx.parser.astext(mod)
    new_mod = rx.transform.shape_lower(mod)
    code = rx.parser.astext(new_mod)
    target = tvm.target.Target("llvm")
    target_host = tvm.target.Target("llvm")
    ex, lib = rx.vm.build(new_mod, target, target_host)
    vm = rx.VirtualMachine(ex, tvm.cpu(), mod=lib)

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3


if __name__ == "__main__":
    test_vm_execute()
    test_vm_multiple_func()
    test_vm_checker()
    test_vm_formalize()
    test_vm_operand()
    test_vm_serialize()
    test_vm_constant_serialize()
    test_vm_shapeof()
    test_vm_storage()
    test_vm_compile_stage0()
    test_vm_compile_stage1()
    test_vm_compile_stage2()
